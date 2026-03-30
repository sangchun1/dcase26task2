import torch
from torch import Tensor, nn
from typing import Optional


def temporal_pooling_from_sequence(
    feats: Tensor,
    num_freq_patches: int = 8,
    strict: bool = True,
) -> Tensor:
    """
    [B, L, C] 형태의 백본 출력에 대해 temporal pooling을 수행하는 함수.

    기본 사용 가정 (EAT 기반):
        - L = T * F_patch
        - F_patch = num_freq_patches (예: 8)
        - feats: [B, T * F_patch, C]
        - reshape: [B, T, F_patch, C]
        - 시간축 T 에 대해 mean: [B, F_patch, C]
        - F_patch * C 로 flatten: [B, F_patch * C]

    Args:
        feats:
            백본 출력 텐서, 크기 [B, L, C].
        num_freq_patches:
            주파수 패치 개수 F_patch (예: 8).
        strict:
            True 인 경우, L 이 num_freq_patches 로 나누어떨어지지 않으면 에러를 발생.
            False 인 경우, 나머지 토큰을 잘라내어 (L_trunc % num_freq_patches == 0) 이 되도록 맞춤.

    Returns:
        크기 [B, num_freq_patches * C] 의 텐서.
    """
    if feats.ndim != 3:
        raise ValueError(f"feats 는 [B, L, C] (3차원) 이어야 합니다. 현재 shape={feats.shape}")

    B, L, C = feats.shape

    if L % num_freq_patches != 0:
        if strict:
            raise ValueError(
                f"L ({L}) 값이 num_freq_patches ({num_freq_patches}) 로 나누어떨어지지 않습니다."
            )
        # strict=False 인 경우, 나머지 토큰을 잘라내고 reshape 가능하게 맞춤
        valid_L = (L // num_freq_patches) * num_freq_patches
        feats = feats[:, :valid_L, :]
        L = valid_L

    T_patches = L // num_freq_patches  # T

    # [B, L, C] -> [B, T, F, C]
    feats = feats.view(B, T_patches, num_freq_patches, C)

    # 시간축 T 에 대해 평균: [B, T, F, C] -> [B, F, C]
    feats = feats.mean(dim=1)

    # [B, F, C] -> [B, F * C]
    pooled = feats.reshape(B, num_freq_patches * C)

    return pooled


def temporal_pooling_4d(
    feats: Tensor,
    time_dim: int = 2,
) -> Tensor:
    """
    4차원 텐서에 대해 temporal pooling 을 수행하는 보조 함수.

    예: 백본이 직접 [B, F, T, C] 또는 [B, T, F, C] 형태로 출력을 줄 때 사용 가능.

    Args:
        feats:
            4차원 텐서. 구체적인 축 의미는 호출하는 쪽에서 관리.
        time_dim:
            시간 축에 해당하는 차원 인덱스.

    Returns:
        시간 평균 및 flatten 이후의 [B, F * C] 텐서.
    """
    if feats.ndim != 4:
        raise ValueError(f"feats 는 4차원이어야 합니다. 현재 shape={feats.shape}")

    # time_dim 을 2번 축으로 옮겨서 [B, F, T, C] 형태로 맞추기
    if time_dim != 2:
        perm = list(range(feats.ndim))
        perm.pop(time_dim)
        perm.insert(2, time_dim)
        feats = feats.permute(*perm)

    B, F, T, C = feats.shape

    # 시간축 T 에 대해 평균: [B, F, T, C] -> [B, F, C]
    feats = feats.mean(dim=2)

    # [B, F, C] -> [B, F * C]
    pooled = feats.reshape(B, F * C)

    return pooled


def temporal_rdp_pooling_from_sequence(
    feats: Tensor,
    num_freq_patches: int = 8,
    gamma: float = 1.0,
    eps: float = 1e-8,
    strict: bool = False,
) -> Tensor:
    """
    [B, L, C] 형태의 백본 출력에 대해 temporal RDP pooling을 수행하는 함수.

    기본 사용 가정:
        - L = T * F_patch
        - F_patch = num_freq_patches (예: 8)
        - feats: [B, T * F_patch, C]
        - reshape: [B, T, F_patch, C]
        - T 축 평균(mu) 계산
        - 각 시점 deviation d_t = ||x_t - mu||_2 계산
        - sequence 내부 max로 relative deviation 정규화
        - (1 + d_hat)^gamma 가중치 기반 weighted mean
        - [B, F_patch, C] -> [B, F_patch * C]

    Args:
        feats:
            백본 출력 텐서, 크기 [B, L, C].
        num_freq_patches:
            주파수 패치 개수 F_patch (예: 8).
        gamma:
            deviation 강조 강도. gamma=0 이면 mean pooling과 동일.
        eps:
            0으로 나누는 문제를 막는 안정화 상수.
        strict:
            True 인 경우, L 이 num_freq_patches 로 나누어떨어지지 않으면 에러를 발생.
            False 인 경우, 나머지 토큰을 잘라내어 reshape 가능하게 맞춤.

    Returns:
        크기 [B, num_freq_patches * C] 의 텐서.
    """
    if feats.ndim != 3:
        raise ValueError(f"feats 는 [B, L, C] (3차원) 이어야 합니다. 현재 shape={feats.shape}")

    B, L, C = feats.shape
    F = num_freq_patches

    if L % F != 0:
        if strict:
            raise ValueError(
                f"L ({L}) 값이 num_freq_patches ({F}) 로 나누어떨어지지 않습니다."
            )
        valid_L = (L // F) * F
        feats = feats[:, :valid_L, :]
        L = valid_L

    T = L // F
    X = feats.view(B, T, F, C)

    # T 축 평균
    mu = X.mean(dim=1, keepdim=True)  # [B, 1, F, C]

    # (b, t, f)별 D축 L2 deviation
    d = torch.norm(X - mu, p=2, dim=-1)  # [B, T, F]
    d_max = d.max(dim=1, keepdim=True).values  # [B, 1, F]
    d_hat = d / (d_max + eps)

    # T 축 가중치
    w_raw = (1.0 + d_hat).pow(gamma)
    w = w_raw / (w_raw.sum(dim=1, keepdim=True) + eps)  # [B, T, F]

    # weighted temporal sum
    pooled = (w.unsqueeze(-1) * X).sum(dim=1)  # [B, F, C]
    return pooled.reshape(B, F * C)


class TemporalPoolingMLP(nn.Module):
    """
    temporal pooling 이후의 임베딩(예: [B, 6144])을
    중간 차원(mid_dim)을 거쳐 최종 768 차원으로 투영하는 MLP 블록.

    기본 형태:
        6144 -> mid_dim -> 768

    mid_dim 값은 추후 args 에서 받아서 생성자에 넘겨주는 방식으로 사용할 것을 가정한다.
    """

    def __init__(
        self,
        input_dim: int = 6144,
        mid_dim: int = 2048,
        output_dim: int = 512,
        tem_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim
        self.tem_dropout = tem_dropout

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(self.tem_dropout),
            nn.Linear(mid_dim, output_dim),
        )


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, input_dim] 형태의 임베딩 (예: temporal pooling 결과)

        Returns:
            [B, output_dim] 형태의 투영된 임베딩 (예: 768 차원)
        """
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"입력 x 는 [B, {self.input_dim}] 이어야 합니다. 현재 shape={x.shape}"
            )
        return self.mlp(x)


class TemporalClassicAttention(nn.Module):
    """
    시간 축 방향(attention over T)에 대한 클래식 attention 모듈.

    입력:  [B, T, C]
    출력:  [B, T] (시간축에 대한 softmax weight)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout

        self.lin_proj = nn.Linear(input_dim, embed_dim)
        self.v = nn.Parameter(torch.randn(embed_dim))
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: [B, T, C]

        Returns:
            attention_weights: [B, T]
        """
        if inputs.ndim != 3:
            raise ValueError(f"inputs 는 [B, T, C] 이어야 합니다. 현재 shape={inputs.shape}")

        lin_out = self.lin_proj(inputs)  # [B, T, embed_dim]
        lin_out = self.dropout(lin_out)

        # v: [embed_dim] -> [B, embed_dim, 1]
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), -1).unsqueeze(2)
        # [B, T, embed_dim] x [B, embed_dim, 1] -> [B, T, 1] -> [B, T]
        attn_raw = torch.tanh(lin_out.bmm(v_view).squeeze(-1))
        attn_weights = torch.softmax(attn_raw, dim=1)
        return attn_weights


class TemporalAttentivePoolingFromSequence(nn.Module):
    """
    [B, L, C] 형태의 시퀀스에서
    - L = T * F_patch (예: T=64, F_patch=8) 구조를 가정하고,
    - 각 주파수 패치별로 시간축에 대한 attention-weighted mean 을 계산한 뒤,
    - [B, F_patch * C] 임베딩을 반환하는 모듈.

    설계 개념은 `make_md/temporal/add.md` 에 정리한 tem_att_pool 아이디어를 따름.
    """

    def __init__(
        self,
        num_freq_patches: int = 8,
        input_dim: int = 768,
        attn_embed_dim: int = 768,
        strict: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_freq_patches = num_freq_patches
        self.input_dim = input_dim
        self.attn_embed_dim = attn_embed_dim
        self.strict = strict

        self.attention = TemporalClassicAttention(
            input_dim=input_dim,
            embed_dim=attn_embed_dim,
            attn_dropout=attn_dropout,
        )

    def forward(self, feats: Tensor) -> Tensor:
        """
        Args:
            feats: [B, L, C] 형태의 백본 출력 시퀀스
                   (L = T * num_freq_patches 구조 가정)

        Returns:
            pooled: [B, num_freq_patches * C] 형태의 임베딩
        """
        if feats.ndim != 3:
            raise ValueError(f"feats 는 [B, L, C] 이어야 합니다. 현재 shape={feats.shape}")

        B, L, C = feats.shape

        if C != self.input_dim:
            raise ValueError(
                f"채널 차원 C ({C}) 이 설정된 input_dim ({self.input_dim}) 과 다릅니다."
            )

        if L % self.num_freq_patches != 0:
            if self.strict:
                raise ValueError(
                    f"L ({L}) 값이 num_freq_patches ({self.num_freq_patches}) 로 "
                    "나누어떨어지지 않습니다."
                )
            # strict=False 인 경우, 나머지 토큰을 잘라내고 reshape 가능하게 맞춤
            valid_L = (L // self.num_freq_patches) * self.num_freq_patches
            feats = feats[:, :valid_L, :]
            B, L, C = feats.shape

        T_patches = L // self.num_freq_patches  # 시간 패치 개수 T

        # [B, L, C] -> [B, T, F, C]
        feats = feats.view(B, T_patches, self.num_freq_patches, C)
        # [B, T, F, C] -> [B, F, T, C]
        feats = feats.permute(0, 2, 1, 3).contiguous()

        # [B, F, T, C] -> [B * F, T, C]
        feats_2d = feats.view(B * self.num_freq_patches, T_patches, C)

        # 시간축 T 에 대한 attention weight 계산: [B * F, T]
        attn_weights = self.attention(feats_2d)

        # weighted mean: sum_t alpha_t * x_t
        # attn_weights: [B * F, T] -> [B * F, T, 1]
        attn_weights_exp = attn_weights.unsqueeze(-1)
        # [B * F, T, C]
        weighted_feats = feats_2d * attn_weights_exp
        # 시간축 합: [B * F, C]
        pooled_per_freq = weighted_feats.sum(dim=1)

        # [B * F, C] -> [B, F, C]
        pooled_per_freq = pooled_per_freq.view(B, self.num_freq_patches, C)

        # [B, F, C] -> [B, F * C]
        pooled = pooled_per_freq.reshape(B, self.num_freq_patches * C)
        return pooled


