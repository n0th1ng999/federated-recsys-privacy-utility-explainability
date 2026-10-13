import torch
import torch.nn as nn
import torch.nn.functional as F

from support.feature_contract import (
    PAIR_CONTEXT_FEATURE_NAMES,
    PURCHASE_FAMILY_INTENT_WEIGHTS,
    PURCHASE_LOGIT_SCALE,
    PURCHASE_LOGIT_WEIGHTS,
    PURCHASE_PRIOR_SUPPORT_WEIGHTS,
    PURCHASE_PRODUCT_FIT_WEIGHTS,
    PURCHASE_REPLENISHMENT_INTENT_WEIGHTS,
    USER_STATE_INTENT_SUMMARY_FEATURES,
    USER_STATE_OWNERSHIP_SUMMARY_FEATURES,
)


def _feature_index_map(names):
    return {str(name): idx for idx, name in enumerate(names)}


_INTENT_SUMMARY_IDX = _feature_index_map(USER_STATE_INTENT_SUMMARY_FEATURES)
_OWNERSHIP_SUMMARY_IDX = _feature_index_map(USER_STATE_OWNERSHIP_SUMMARY_FEATURES)
_PAIR_CONTEXT_IDX = _feature_index_map(PAIR_CONTEXT_FEATURE_NAMES)


class SequenceAttentionPool(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        if x.dim() != 3:
            raise ValueError(f"Expected [batch, seq, dim], got {tuple(x.shape)}")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights = weights / denom
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        pooled = torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
        return pooled


class SubBranchSequenceEncoder(nn.Module):
    def __init__(self, cluster_dim, summary_dim, latent_dim=128, dropout=0.1):
        super().__init__()
        self.cluster_proj = nn.Sequential(
            nn.Linear(cluster_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(summary_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.pool = SequenceAttentionPool(latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.support_head = nn.Linear(summary_dim + latent_dim, 1)

    def forward(self, summary_x, cluster_values, cluster_mask=None):
        summary_latent = self.summary_proj(summary_x)
        if cluster_values.numel() == 0:
            pooled = torch.zeros_like(summary_latent)
        else:
            if cluster_mask is None:
                cluster_mask = cluster_values.abs().sum(dim=-1) > 0
            cluster_latent = self.cluster_proj(cluster_values)
            pooled = self.pool(cluster_latent, cluster_mask)
        combined = self.norm(summary_latent + pooled)
        support = torch.sigmoid(self.support_head(torch.cat([summary_x, combined], dim=-1)))
        return combined, support


class UserStateBranchEncoder(nn.Module):
    def __init__(self, dims, latent_dim=128, dropout=0.1):
        super().__init__()
        self.intent_summary_dim = dims["intent_summary_dim"]
        self.ownership_summary_dim = dims["ownership_summary_dim"]
        self.intent_encoder = SubBranchSequenceEncoder(
            dims["intent_cluster_dim"],
            dims["intent_summary_dim"],
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.ownership_encoder = SubBranchSequenceEncoder(
            dims["ownership_cluster_dim"],
            dims["ownership_summary_dim"],
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.shape_gate = nn.Sequential(
            nn.Linear(dims["intent_summary_dim"] + dims["ownership_summary_dim"] + latent_dim * 2, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.purchase_norm = nn.LayerNorm(latent_dim)
        self.rank_norm = nn.LayerNorm(latent_dim)
        self.purchase_leak_gate = nn.Sequential(
            nn.Linear(dims["intent_summary_dim"] + dims["ownership_summary_dim"] + latent_dim * 2, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        self.rank_context_gate = nn.Sequential(
            nn.Linear(dims["intent_summary_dim"] + dims["ownership_summary_dim"] + latent_dim * 2, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        self.strength_head = nn.Linear(dims["intent_summary_dim"] + dims["ownership_summary_dim"] + latent_dim * 2, 1)

    def forward(self, branch_inputs):
        intent_summary = branch_inputs["intent_summary"]
        ownership_summary = branch_inputs["ownership_summary"]
        intent_clusters = branch_inputs["intent_clusters"]
        ownership_clusters = branch_inputs["ownership_clusters"]

        intent_latent, intent_support = self.intent_encoder(intent_summary, intent_clusters)
        ownership_latent, ownership_support = self.ownership_encoder(ownership_summary, ownership_clusters)

        shape_in = torch.cat([intent_summary, ownership_summary, intent_latent, ownership_latent], dim=-1)
        ownership_shape_strength = torch.sigmoid(self.shape_gate(shape_in))
        user_state = self.output_norm(intent_latent + ownership_shape_strength * ownership_latent)
        purchase_leak_strength = 0.20 * torch.sigmoid(self.purchase_leak_gate(shape_in))
        rank_context_strength = 0.10 * torch.sigmoid(self.rank_context_gate(shape_in))
        purchase_latent = self.purchase_norm(intent_latent + purchase_leak_strength * ownership_latent)
        rank_latent = self.rank_norm(ownership_latent + rank_context_strength * intent_latent)
        strength_in = torch.cat([intent_summary, ownership_summary, intent_latent, ownership_latent], dim=-1)
        strength = torch.sigmoid(self.strength_head(strength_in))
        diagnostics = {
            "intent_support": intent_support,
            "ownership_support": ownership_support,
            "ownership_shape_strength": ownership_shape_strength,
            "user_state_support": strength,
            "purchase_leak_strength": purchase_leak_strength,
            "rank_context_strength": rank_context_strength,
            "purchase_latent": purchase_latent,
            "rank_latent": rank_latent,
            "intent_latent": intent_latent,
            "ownership_latent": ownership_latent,
        }
        return user_state, diagnostics


class CollaborativeBranchEncoder(nn.Module):
    def __init__(self, dims, latent_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = SubBranchSequenceEncoder(
            dims["cluster_dim"],
            dims["channel_support_dim"],
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def forward(self, branch_inputs):
        channel_support = branch_inputs["channel_support"]
        clusters = branch_inputs["clusters"]
        collab, support = self.encoder(channel_support, clusters)
        diagnostics = {
            "collaborative_support": support,
        }
        return collab, diagnostics


class PairContextEncoder(nn.Module):
    def __init__(self, dims, latent_dim=128, dropout=0.1):
        super().__init__()
        self.context_dim = dims["context_dim"]
        self.proj = nn.Sequential(
            nn.Linear(self.context_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.support_head = nn.Linear(self.context_dim + latent_dim, 1)

    def forward(self, branch_inputs):
        context = branch_inputs["context"]
        latent = self.norm(self.proj(context))
        support = torch.sigmoid(self.support_head(torch.cat([context, latent], dim=-1)))
        diagnostics = {
            "pair_context_support": support,
        }
        return latent, diagnostics


class CoreHybridRecSys(nn.Module):
    def __init__(self, branch_dims, latent_dim=128, fusion_dim=128, dropout=0.1, num_heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.branch_dims = branch_dims
        self.reg_scalar_dim = 7
        self.family_pair_feature_names = [
            "intent_score",
            "source_intent_score",
            "family_local_intent",
            "source_purchase_recency",
            "source_family_recovery",
            "expected_cycle_norm",
            "user_cycle_ratio",
            "user_faster_than_expected",
            "user_slower_than_expected",
            "same_item_penalty",
            "substitute_penalty",
            "cluster_suppression",
            "intent_momentum",
            "interaction_boost",
        ]
        self.product_fit_feature_names = [
            "brand_match",
            "keyword_match",
            "size_match",
            "color_match",
            "material_match",
            "source_purchase_score",
            "source_budget_alignment",
            "global_budget_alignment",
            "price_norm",
        ]
        self.branch_order = ["user_state", "collaborative", "pair_context"]
        self.user_state_branch = UserStateBranchEncoder(branch_dims["user_state"], latent_dim=latent_dim, dropout=dropout)
        self.collaborative_branch = CollaborativeBranchEncoder(branch_dims["collaborative"], latent_dim=latent_dim, dropout=dropout)
        self.pair_context_branch = PairContextEncoder(branch_dims["pair_context"], latent_dim=latent_dim, dropout=dropout)
        family_in_dim = latent_dim + 2 + len(self.family_pair_feature_names)
        product_fit_in_dim = len(self.product_fit_feature_names)
        rank_context_in_dim = latent_dim + branch_dims["user_state"]["ownership_summary_dim"] + 2
        self.family_intent_encoder = nn.Sequential(
            nn.Linear(family_in_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.product_fit_encoder = nn.Sequential(
            nn.Linear(product_fit_in_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.rank_context_encoder = nn.Sequential(
            nn.Linear(rank_context_in_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.branch_confidence = nn.ModuleDict({
            "user_state": nn.Linear(latent_dim, 1),
            "collaborative": nn.Linear(latent_dim, 1),
            "pair_context": nn.Linear(latent_dim, 1),
        })
        gate_in_dim = latent_dim * 3
        self.personalization_gate = nn.Sequential(
            nn.Linear(gate_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.rank_personalization_gate = nn.Sequential(
            nn.Linear(gate_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.fusion_proj = nn.Linear(latent_dim * 3, fusion_dim)
        self.fc1 = nn.Linear(fusion_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.residual_proj = nn.Linear(fusion_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        reg_input_dim = latent_dim * 3 + self.reg_scalar_dim
        self.reg_fc1 = nn.Linear(reg_input_dim, latent_dim)
        self.reg_fc2 = nn.Linear(latent_dim, latent_dim)
        self.reg_residual = nn.Linear(reg_input_dim, latent_dim)
        self.reg_head = nn.Linear(latent_dim, 1)
        self.rank_head = nn.Linear(latent_dim, 1)
        self.conf_head = nn.Linear(latent_dim, 1)

    def _col(self, tensor, idx_map, name):
        idx = idx_map[name]
        return torch.clamp(tensor[:, idx:idx + 1], 0.0, 1.0)

    def _stack_pair_cols(self, pair_context, names):
        return torch.cat([self._col(pair_context, _PAIR_CONTEXT_IDX, name) for name in names], dim=-1)

    def _purchase_components(self, user_state_inputs, hist_diag, pair_context_inputs):
        intent_summary = user_state_inputs["intent_summary"]
        ownership_summary = user_state_inputs["ownership_summary"]
        pair_context = pair_context_inputs["context"]

        interest_strength = self._col(intent_summary, _INTENT_SUMMARY_IDX, "interest_strength")
        source_view_share = self._col(intent_summary, _INTENT_SUMMARY_IDX, "source_view_share")
        active_view_mass = self._col(intent_summary, _INTENT_SUMMARY_IDX, "active_view_mass")
        active_wishlist_mass = self._col(intent_summary, _INTENT_SUMMARY_IDX, "active_wishlist_mass")
        active_cart_mass = self._col(intent_summary, _INTENT_SUMMARY_IDX, "active_cart_mass")
        focus_concentration = self._col(intent_summary, _INTENT_SUMMARY_IDX, "focus_concentration")
        ownership_interest_ratio = self._col(ownership_summary, _OWNERSHIP_SUMMARY_IDX, "ownership_interest_ratio")
        budget_coverage = self._col(ownership_summary, _OWNERSHIP_SUMMARY_IDX, "budget_coverage")
        budget_global_norm = self._col(ownership_summary, _OWNERSHIP_SUMMARY_IDX, "budget_global_norm")

        intent_score = self._col(pair_context, _PAIR_CONTEXT_IDX, "intent_score")
        source_intent_score = self._col(pair_context, _PAIR_CONTEXT_IDX, "source_intent_score")
        source_purchase_recency = self._col(pair_context, _PAIR_CONTEXT_IDX, "source_purchase_recency")
        source_family_recovery = self._col(pair_context, _PAIR_CONTEXT_IDX, "source_family_recovery")
        expected_cycle_norm = self._col(pair_context, _PAIR_CONTEXT_IDX, "expected_cycle_norm")
        user_cycle_ratio = self._col(pair_context, _PAIR_CONTEXT_IDX, "user_cycle_ratio")
        user_faster_than_expected = self._col(pair_context, _PAIR_CONTEXT_IDX, "user_faster_than_expected")
        user_slower_than_expected = self._col(pair_context, _PAIR_CONTEXT_IDX, "user_slower_than_expected")
        same_item_penalty = self._col(pair_context, _PAIR_CONTEXT_IDX, "same_item_penalty")
        substitute_penalty = self._col(pair_context, _PAIR_CONTEXT_IDX, "substitute_penalty")
        cluster_suppression = self._col(pair_context, _PAIR_CONTEXT_IDX, "cluster_suppression")
        source_purchase_score = self._col(pair_context, _PAIR_CONTEXT_IDX, "source_purchase_score")
        source_budget_alignment = self._col(pair_context, _PAIR_CONTEXT_IDX, "source_budget_alignment")
        global_budget_alignment = self._col(pair_context, _PAIR_CONTEXT_IDX, "global_budget_alignment")
        price_norm = self._col(pair_context, _PAIR_CONTEXT_IDX, "price_norm")
        intent_momentum = self._col(pair_context, _PAIR_CONTEXT_IDX, "intent_momentum")
        interaction_boost = self._col(pair_context, _PAIR_CONTEXT_IDX, "interaction_boost")

        behavioral_intent = (
            0.28 * interest_strength
            + 0.16 * source_view_share
            + 0.14 * active_view_mass
            + 0.18 * active_wishlist_mass
            + 0.26 * active_cart_mass
            + 0.08 * focus_concentration
        )
        replenishment_intent = (
            PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["source_family_recovery"] * source_family_recovery
            + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["source_purchase_recency"] * source_purchase_recency
            + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["expected_cycle_norm"] * expected_cycle_norm
            + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["user_faster_than_expected"] * user_faster_than_expected
            + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["user_slower_than_expected"] * user_slower_than_expected
            + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["user_cycle_ratio"] * user_cycle_ratio
        )
        family_intent = torch.clamp(
            PURCHASE_FAMILY_INTENT_WEIGHTS["behavioral_intent"] * behavioral_intent
            + PURCHASE_FAMILY_INTENT_WEIGHTS["intent_score"] * intent_score
            + PURCHASE_FAMILY_INTENT_WEIGHTS["source_intent_score"] * source_intent_score
            + PURCHASE_FAMILY_INTENT_WEIGHTS["intent_momentum"] * intent_momentum
            + PURCHASE_FAMILY_INTENT_WEIGHTS["interaction_boost"] * interaction_boost
            + PURCHASE_FAMILY_INTENT_WEIGHTS["replenishment_intent"] * replenishment_intent
            + PURCHASE_FAMILY_INTENT_WEIGHTS["same_item_relief"] * (1.0 - same_item_penalty)
            + PURCHASE_FAMILY_INTENT_WEIGHTS["substitute_relief"] * (1.0 - substitute_penalty)
            + PURCHASE_FAMILY_INTENT_WEIGHTS["cluster_relief"] * (1.0 - cluster_suppression)
            + 0.04 * ownership_interest_ratio
            + 0.03 * budget_coverage,
            0.0,
            1.0,
        )
        product_fit = torch.clamp(
            PURCHASE_PRODUCT_FIT_WEIGHTS["brand_match"] * self._col(pair_context, _PAIR_CONTEXT_IDX, "brand_match")
            + PURCHASE_PRODUCT_FIT_WEIGHTS["keyword_match"] * self._col(pair_context, _PAIR_CONTEXT_IDX, "keyword_match")
            + PURCHASE_PRODUCT_FIT_WEIGHTS["size_match"] * self._col(pair_context, _PAIR_CONTEXT_IDX, "size_match")
            + PURCHASE_PRODUCT_FIT_WEIGHTS["color_match"] * self._col(pair_context, _PAIR_CONTEXT_IDX, "color_match")
            + PURCHASE_PRODUCT_FIT_WEIGHTS["material_match"] * self._col(pair_context, _PAIR_CONTEXT_IDX, "material_match")
            + PURCHASE_PRODUCT_FIT_WEIGHTS["source_purchase_score"] * source_purchase_score
            + PURCHASE_PRODUCT_FIT_WEIGHTS["source_budget_alignment"] * source_budget_alignment
            + PURCHASE_PRODUCT_FIT_WEIGHTS["global_budget_alignment"] * global_budget_alignment
            + PURCHASE_PRODUCT_FIT_WEIGHTS["price_norm"] * price_norm,
            0.0,
            1.0,
        )
        family_primitives = self._stack_pair_cols(pair_context, self.family_pair_feature_names)
        product_primitives = self._stack_pair_cols(pair_context, self.product_fit_feature_names)
        ownership_budget_features = torch.cat([ownership_interest_ratio, budget_coverage], dim=-1)
        family_latent = self.family_intent_encoder(
            torch.cat([hist_diag["intent_latent"], ownership_budget_features, family_primitives], dim=-1)
        )
        product_latent = self.product_fit_encoder(product_primitives)
        rank_context_latent = self.rank_context_encoder(
            torch.cat(
                [
                    hist_diag["rank_latent"],
                    ownership_summary,
                    self._stack_pair_cols(pair_context, ["source_purchase_recency", "cluster_suppression"]),
                ],
                dim=-1,
            )
        )
        return {
            "family_intent": family_intent,
            "product_fit": product_fit,
            "family_latent": family_latent,
            "product_latent": product_latent,
            "rank_context_latent": rank_context_latent,
            "budget_global_norm": budget_global_norm,
        }

    def _purchase_prior(self, hist_diag, pair_diag, components):
        hist_support = hist_diag["user_state_support"]
        intent_support = hist_diag["intent_support"]
        ownership_support = hist_diag["ownership_support"]
        pair_support = pair_diag["pair_context_support"]
        family_intent = components["family_intent"]
        product_fit = components["product_fit"]
        intent_fit_synergy = family_intent * product_fit
        intent_fit_compound = intent_fit_synergy * (0.5 * (family_intent + product_fit))

        base_logit = (
            PURCHASE_LOGIT_WEIGHTS["bias"]
            + PURCHASE_LOGIT_WEIGHTS["family_intent"] * family_intent
            + PURCHASE_LOGIT_WEIGHTS["product_fit"] * product_fit
            + PURCHASE_LOGIT_WEIGHTS["intent_fit_synergy"] * intent_fit_synergy
            + PURCHASE_LOGIT_WEIGHTS["intent_fit_compound"] * intent_fit_compound
        )
        logit = (
            PURCHASE_LOGIT_SCALE * base_logit
            + PURCHASE_PRIOR_SUPPORT_WEIGHTS["hist_support"] * hist_support
            + PURCHASE_PRIOR_SUPPORT_WEIGHTS["pair_support"] * pair_support
            + PURCHASE_PRIOR_SUPPORT_WEIGHTS["intent_support"] * intent_support
            + PURCHASE_PRIOR_SUPPORT_WEIGHTS["ownership_support"] * ownership_support
        )
        return torch.clamp(torch.sigmoid(logit), 0.03, 0.97)

    def _rank_prior(self, hist_diag, coll_diag, pair_diag):
        hist_support = hist_diag["user_state_support"]
        ownership_support = hist_diag["ownership_support"]
        ownership_shape_strength = hist_diag["ownership_shape_strength"]
        rank_context_strength = hist_diag["rank_context_strength"]
        coll_support = coll_diag["collaborative_support"]
        pair_support = pair_diag["pair_context_support"]
        prior = (
            0.54
            + 0.18 * hist_support
            + 0.24 * ownership_support
            + 0.10 * ownership_shape_strength
            + 0.04 * rank_context_strength
            + 0.10 * pair_support
            - 0.08 * coll_support
        )
        return torch.clamp(prior, 0.52, 0.98)

    def _forward_impl(self, inputs, masks=None):
        user_state_inputs = inputs["user_state"]
        collaborative_inputs = inputs["collaborative"]
        pair_context_inputs = inputs["pair_context"]

        user_state_out, user_state_diag = self.user_state_branch(user_state_inputs)
        collaborative_out, collaborative_diag = self.collaborative_branch(collaborative_inputs)
        pair_context_out, pair_context_diag = self.pair_context_branch(pair_context_inputs)
        intent_support = user_state_diag["intent_support"]
        ownership_support = user_state_diag["ownership_support"]
        purchase_components = self._purchase_components(user_state_inputs, user_state_diag, pair_context_inputs)
        family_intent = purchase_components["family_intent"]
        product_fit = purchase_components["product_fit"]
        family_latent = purchase_components["family_latent"]
        product_latent = purchase_components["product_latent"]
        rank_context_latent = purchase_components["rank_context_latent"]

        user_state_conf = torch.sigmoid(self.branch_confidence["user_state"](user_state_out))
        collaborative_conf = torch.sigmoid(self.branch_confidence["collaborative"](collaborative_out))
        pair_context_conf = torch.sigmoid(self.branch_confidence["pair_context"](pair_context_out))
        purchase_hist_conf = torch.clamp(0.55 * user_state_conf + 0.45 * intent_support, 0.0, 1.0)
        rank_hist_conf = torch.clamp(0.55 * user_state_conf + 0.45 * ownership_support, 0.0, 1.0)

        purchase_prior = self._purchase_prior(
            user_state_diag,
            pair_context_diag,
            purchase_components,
        )
        rank_prior = self._rank_prior(user_state_diag, collaborative_diag, pair_context_diag)

        purchase_gate_features = torch.cat([
            family_latent,
            collaborative_out,
            product_latent,
        ], dim=-1)
        rank_gate_features = torch.cat([
            rank_context_latent,
            collaborative_out,
            pair_context_out,
        ], dim=-1)
        purchase_gate_residual = 0.35 * torch.tanh(self.personalization_gate(purchase_gate_features))
        rank_gate_residual = 0.25 * torch.tanh(self.rank_personalization_gate(rank_gate_features))
        purchase_strength = torch.sigmoid(
            torch.logit(purchase_prior.clamp(1e-4, 1 - 1e-4)) + purchase_gate_residual
        )
        rank_strength = torch.sigmoid(
            torch.logit(rank_prior.clamp(1e-4, 1 - 1e-4)) + rank_gate_residual
        )

        purchase_hist_weighted = family_latent * purchase_hist_conf * purchase_strength
        purchase_coll_weighted = collaborative_out * collaborative_conf * (1.0 - purchase_strength)
        purchase_pair_weighted = product_latent * pair_context_conf
        rank_hist_weighted = rank_context_latent * rank_hist_conf * rank_strength
        rank_coll_weighted = collaborative_out * collaborative_conf * (1.0 - rank_strength)
        rank_pair_weighted = pair_context_out * pair_context_conf
        reg_scalar_features = torch.cat(
            [
                purchase_strength,
                purchase_hist_conf,
                collaborative_conf,
                pair_context_conf,
                family_intent,
                product_fit,
                family_intent * product_fit,
            ],
            dim=-1,
        )
        reg_features = torch.cat([purchase_hist_weighted, purchase_coll_weighted, purchase_pair_weighted, reg_scalar_features], dim=-1)
        reg_x1 = self.dropout(F.relu(self.reg_fc1(reg_features)))
        reg_x2 = self.dropout(F.relu(self.reg_fc2(reg_x1)))
        reg_latent = reg_x2 + self.reg_residual(reg_features)

        fused = self.fusion_proj(torch.cat([rank_hist_weighted, rank_coll_weighted, rank_pair_weighted], dim=-1))

        x1 = self.dropout(F.relu(self.fc1(fused)))
        x2 = self.dropout(F.relu(self.fc2(x1)))
        x = self.output_norm(x2 + self.residual_proj(fused))

        purchase_logit = self.reg_head(reg_latent).view(-1)
        rank_score = self.rank_head(x).view(-1)
        confidence_logit = self.conf_head(x).view(-1)
        return purchase_logit, rank_score, confidence_logit

    def forward(self, inputs, masks=None):
        return self._forward_impl(inputs, masks=masks)
