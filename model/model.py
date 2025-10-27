import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def get_activation(activation_str):
    if activation_str == 'relu':
        return F.relu
    elif activation_str == 'leaky_relu':
        return F.leaky_relu
    elif activation_str == 'sigmoid':
        return torch.sigmoid
    elif activation_str == 'tanh':
        return torch.tanh
    elif activation_str == 'none':
        return None
    else:
        raise ValueError(f"Unsupported activation: {activation_str}")


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True, drop=None):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.drop = drop
        if drop is not None:
            self.dropout = nn.Dropout(p=drop)
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        if self.drop is not None:
            x = self.dropout(x)
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    def __init__(self, D, args, drop=None):
        super(STEmbedding, self).__init__()
        self.D = D
        se_bn_decay = 0.1
        se_act1 = 'relu'
        se_act2 = 'none'
        se_drop = 0.01

        te_bn_decay = 0.1
        te_act1 = 'relu'
        te_act2 = 'none'
        te_drop = 0.01

        self.FC_se = FC(
            input_dims=[64, D],
            units=[D, D],
            activations=[get_activation(se_act1), get_activation(se_act2)],
            bn_decay=se_bn_decay,
            drop=se_drop
        )

        self.FC_te = nn.Sequential(
            FC(
                input_dims=[3, D],
                units=[D, D],
                activations=[get_activation(te_act1), get_activation(te_act2)],
                bn_decay=te_bn_decay,
                drop=te_drop
            ),
            FC(
                input_dims=[D, D],
                units=[D, D],
                activations=[get_activation(te_act1), get_activation(te_act2)],
                bn_decay=te_bn_decay,
                drop=te_drop
            )
        )

    def forward(self, SE, TE):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        TE = TE.unsqueeze(2).float()
        TE = self.FC_te(TE)
        STE = SE + TE
        return STE


class FourierFilter(nn.Module):
    def __init__(self, args, filter_types=['adaptive', 'low', 'high'], drop=None):
        super(FourierFilter, self).__init__()
        self.filter_types = filter_types
        self.num_filters = len(filter_types)
        self.args = args

        fft_bn_decay = 0.1
        fft_act1 ='relu'
        fft_act2 ='none'
        fft_drop =0.01
        if 'adaptive' in filter_types:
            self.adaptive_weights = nn.Parameter(torch.randn(1))
        self.distribution_correction = FC(
            input_dims=[1, 1],
            units=[1, 1],
            activations=[get_activation(fft_act1), get_activation(fft_act2)],
            bn_decay=fft_bn_decay,
            drop=fft_drop
        )

    def forward(self, x):
        original = x
        batch_size, num_steps, num_nodes, feat_dim = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        freqs = torch.fft.rfftfreq(num_steps, d=1 / num_steps, device=x.device)
        filtered_results = []

        if 'adaptive' in self.filter_types:
            adaptive_filter = torch.sigmoid(self.adaptive_weights)
            adaptive_filter = adaptive_filter.view(1, -1, 1, 1)
            adaptive_result = x_fft * adaptive_filter
            filtered_results.append(torch.fft.irfft(adaptive_result, n=num_steps, dim=1, norm='ortho'))

        if 'low' in self.filter_types:
            cutoff = 0.1 * freqs.max()
            lowpass = torch.where(freqs <= cutoff, 1.0, 0.0).view(1, -1, 1, 1)
            lowpass_result = x_fft * lowpass
            filtered_results.append(torch.fft.irfft(lowpass_result, n=num_steps, dim=1, norm='ortho'))

        if 'high' in self.filter_types:
            cutoff = 0.1 * freqs.max()
            highpass = torch.where(freqs >= cutoff, 1.0, 0.0).view(1, -1, 1, 1)
            highpass_result = x_fft * highpass
            filtered_results.append(torch.fft.irfft(highpass_result, n=num_steps, dim=1, norm='ortho'))

        filtered_results.append(original)
        fused = torch.cat(filtered_results, dim=-1)
        gate = torch.softmax(torch.randn_like(fused), dim=-1)
        fused = torch.sum(fused * gate, dim=-1, keepdim=True)
        corrected = self.distribution_correction(fused)
        mean_original = torch.mean(original, dim=(0, 1, 2), keepdim=True)
        std_original = torch.std(original, dim=(0, 1, 2), keepdim=True)
        mean_corrected = torch.mean(corrected, dim=(0, 1, 2), keepdim=True)
        std_corrected = torch.std(corrected, dim=(0, 1, 2), keepdim=True)
        corrected = (corrected - mean_corrected) / (std_corrected + 1e-8)
        corrected = corrected * std_original + mean_original

        return corrected


class GraphDiffusionConv(nn.Module):
    def __init__(self, adj_matrix, input_dim, output_dim, bn_decay=0.1):
        super(GraphDiffusionConv, self).__init__()
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        self.register_buffer('adj_matrix', adj_matrix)
        self.num_nodes = adj_matrix.shape[0]
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W)
        self.norm_adj = self.normalize_adjacency()
        self.bn = nn.BatchNorm2d(output_dim, momentum=bn_decay)

    def normalize_adjacency(self):
        adj = self.adj_matrix
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def forward(self, X):
        batch_size, num_steps, num_nodes, feat_dim = X.shape
        if self.adj_matrix.device != X.device:
            self.adj_matrix = self.adj_matrix.to(X.device)

        X_reshaped = X.reshape(batch_size * num_steps, num_nodes, feat_dim)
        self.norm_adj = self.norm_adj.to(dtype=torch.float32)
        diffused = torch.einsum('ij,bjk->bik', self.norm_adj, X_reshaped)
        transformed = torch.einsum('bik,kl->bil', diffused, self.W)
        transformed = F.relu(transformed)
        transformed = transformed.reshape(batch_size, num_steps, num_nodes, -1)
        transformed = transformed.permute(0, 3, 1, 2)
        transformed = self.bn(transformed)
        transformed = transformed.permute(0, 2, 3, 1)

        return transformed


class KeyNodeAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, top_k=10, args=None, drop=None):
        super(KeyNodeAttention, self).__init__()
        self.num_heads = num_heads
        self.top_k = top_k
        self.output_dim = output_dim
        w_bn_decay = 0.1
        w_drop = 0.01
        w_act1 = 'relu'
        w_act2 = 'none'
        wq_bn_decay = 0.1
        wq_drop = 0.01
        wq_act = 'none'
        wk_bn_decay = 0.1
        wk_drop = 0.01
        wk_act = 'none'
        wv_bn_decay = 0.1
        wv_drop = 0.01
        wv_act = 'none'

        fusion_bn_decay = 0.1
        fusion_drop = 0.01
        fusion_act = 'relu'

        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.W = FC(
            input_dims=[input_dim * 2, output_dim],
            units=[output_dim, output_dim],
            activations=[get_activation(w_act1), get_activation(w_act2)],
            bn_decay=w_bn_decay,
            drop=w_drop
        )
        self.W_q = FC(
            input_dims=output_dim,
            units=output_dim,
            activations=get_activation(wq_act),
            bn_decay=wq_bn_decay,
            drop=wq_drop
        )
        self.W_k = FC(
            input_dims=output_dim,
            units=output_dim,
            activations=get_activation(wk_act),
            bn_decay=wk_bn_decay,
            drop=wk_drop
        )
        self.W_v = FC(
            input_dims=output_dim,
            units=output_dim,
            activations=get_activation(wv_act),
            bn_decay=wv_bn_decay,
            drop=wv_drop
        )
        self.fusion = FC(
            input_dims=output_dim * 2,
            units=output_dim,
            activations=get_activation(fusion_act),
            bn_decay=fusion_bn_decay,
            drop=fusion_drop
        )

    def forward(self, X, adj_matrix, STE):
        batch_size, num_steps, num_nodes, feat_dim = X.shape
        X = torch.cat((X, STE), dim=-1)
        X = self.W(X)

        reshaped = X.reshape(-1, self.output_dim)
        importance_scores = self.importance_net(reshaped)
        importance_scores = importance_scores.reshape(batch_size, num_steps, num_nodes)
        importance = torch.mean(importance_scores, dim=1)
        importance = torch.mean(importance, dim=0)
        _, top_indices = torch.topk(importance, self.top_k)
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=X.device)
        mask[top_indices] = True
        key_X = X[:, :, mask, :]
        Q = self.W_q(key_X)
        K = self.W_k(key_X)
        V = self.W_v(key_X)
        Q = Q.view(batch_size, num_steps, self.top_k, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        K = K.view(batch_size, num_steps, self.top_k, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        V = V.view(batch_size, num_steps, self.top_k, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).contiguous()
        attn_output = attn_output.view(batch_size, num_steps, self.top_k, -1)
        full_output = torch.zeros(batch_size, num_steps, num_nodes, self.output_dim, device=X.device)
        full_output[:, :, mask, :] = attn_output

        fused = self.fusion(torch.cat([X, full_output], dim=-1))

        return fused


class spatialAttention(nn.Module):
    def __init__(self, K, d, args, drop=None):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K

        q_bn_decay = 0.1
        q_drop = 0.01
        q_act = 'relu'
        k_bn_decay = 0.1
        k_drop = 0.01
        k_act = 'relu'
        v_bn_decay = 0.1
        v_drop = 0.01
        v_act = 'relu'
        fc_bn_decay = 0.1
        fc_drop = 0.01
        fc_act = 'relu'

        self.FC_q = FC(
            input_dims=2 * D,
            units=D,
            activations=get_activation(q_act),
            bn_decay=q_bn_decay,
            drop=q_drop
        )
        self.FC_k = FC(
            input_dims=2 * D,
            units=D,
            activations=get_activation(k_act),
            bn_decay=k_bn_decay,
            drop=k_drop
        )
        self.FC_v = FC(
            input_dims=2 * D,
            units=D,
            activations=get_activation(v_act),
            bn_decay=v_bn_decay,
            drop=v_drop
        )
        self.FC = FC(
            input_dims=D,
            units=D,
            activations=get_activation(fc_act),
            bn_decay=fc_bn_decay,
            drop=fc_drop
        )

    def forward(self, X, STE):
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        return X


class FFTbasedTemporalPeriodExtraction(nn.Module):
    def __init__(self, input_dim, output_dim, num_freq=5, bn_decay=0.1):
        super(FFTbasedTemporalPeriodExtraction, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_freq = num_freq

        self.fourier_transform = nn.Linear(input_dim, num_freq * 2)
        self.freq_processor = nn.Sequential(
            nn.Linear(num_freq * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_freq * 2)
        )
        self.predictor = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm2d(output_dim, momentum=bn_decay)

    def forward(self, X):
        batch_size, num_steps, num_nodes, feat_dim = X.shape
        weights = self.fourier_transform(X)
        weights = weights.view(batch_size, num_steps, num_nodes, self.num_freq, 2)
        weights_mean = torch.mean(weights, dim=1)
        X_reshaped = X.permute(0, 2, 3, 1)
        X_fft = torch.fft.rfft(X_reshaped, dim=-1, norm='ortho')
        num_freqs = X_fft.size(-1)
        selected_real = X_fft.real[..., :self.num_freq]
        selected_imag = X_fft.imag[..., :self.num_freq]
        weights_real = weights_mean[..., 0].unsqueeze(2)
        weights_imag = weights_mean[..., 1].unsqueeze(2)
        weighted_real = selected_real * weights_real
        weighted_imag = selected_imag * weights_imag
        freq_combined = torch.cat([weighted_real, weighted_imag], dim=-1)
        freq_combined = freq_combined.reshape(batch_size * num_nodes * feat_dim, self.num_freq * 2)
        freq_processed = self.freq_processor(freq_combined)
        freq_processed = freq_processed.view(batch_size, num_nodes, feat_dim, self.num_freq * 2)
        new_real, new_imag = torch.split(freq_processed, self.num_freq, dim=-1)
        new_real_full = torch.zeros(batch_size, num_nodes, feat_dim, num_freqs, device=X.device)
        new_imag_full = torch.zeros_like(new_real_full)
        new_real_full[..., :self.num_freq] = new_real
        new_imag_full[..., :self.num_freq] = new_imag

        X_fft_new = torch.complex(new_real_full, new_imag_full)
        X_processed = torch.fft.irfft(X_fft_new, n=num_steps, dim=-1, norm='ortho')
        X_processed = X_processed.permute(0, 3, 1, 2)

        pred = self.predictor(X_processed)
        output = X + pred
        output = output.permute(0, 3, 1, 2)
        output = self.bn(output)
        output = output.permute(0, 2, 3, 1)
        return output


class TemporalAttention(nn.Module):
    def __init__(self, input_dim, output_dim, args, drop=None):
        super(TemporalAttention, self).__init__()

        q_bn_decay = 0.1
        q_drop = 0.01
        q_act ='none'
        k_bn_decay =0.1
        k_drop =0.01
        k_act ='none'
        v_bn_decay =0.1
        v_drop =0.01
        v_act ='none'
        fusion_bn_decay =0.1
        fusion_drop = 0.01
        fusion_act = 'relu'

        self.W_q = FC(
            input_dims=input_dim * 2,
            units=output_dim,
            activations=get_activation(q_act),
            bn_decay=q_bn_decay,
            drop=q_drop
        )
        self.W_k = FC(
            input_dims=input_dim * 2,
            units=output_dim,
            activations=get_activation(k_act),
            bn_decay=k_bn_decay,
            drop=k_drop
        )
        self.W_v = FC(
            input_dims=input_dim * 2,
            units=output_dim,
            activations=get_activation(v_act),
            bn_decay=v_bn_decay,
            drop=v_drop
        )
        self.fusion = FC(
            input_dims=output_dim * 2,
            units=output_dim,
            activations=get_activation(fusion_act),
            bn_decay=fusion_bn_decay,
            drop=fusion_drop
        )

    def forward(self, X, STE, mask_future=True):
        batch_size, num_steps, num_nodes, feat_dim = X.shape
        original_X = X
        X = torch.cat((X, STE), dim=-1)

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        attn_scores = torch.einsum('btnd,bknd->btnk', Q, K) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        if mask_future:
            mask = torch.tril(torch.ones(num_steps, num_steps, device=X.device), diagonal=0)
            mask = mask.view(1, num_steps, 1, num_steps)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.einsum('btnk,bknd->btnd', attn_weights, V)
        fused = self.fusion(torch.cat([original_X, context], dim=-1))
        return fused


class QueryenhancedAttention(nn.Module):
    def __init__(self, output_dim, args, drop=None):
        super(QueryenhancedAttention, self).__init__()
        self.output_dim = output_dim

        
        q_bn_decay = 0.1
        q_drop = 0.01
        q_act = 'none'
        k_bn_decay =0.1
        k_drop =0.01
        k_act = 'none'
        v_bn_decay =0.1
        v_drop =0.01
        v_act ='none'
        map_bn_decay =0.1
        map_drop = 0.01
        map_act = 'relu'
        fusion_bn_decay = 0.1
        fusion_drop = 0.01
        fusion_act = 'relu'
        self.W_q = FC(
            input_dims=output_dim,
            units=output_dim,
            activations=get_activation(q_act),
            bn_decay=q_bn_decay,
            drop=q_drop
        )
        self.W_k = FC(
            input_dims=output_dim,
            units=output_dim,
            activations=get_activation(k_act),
            bn_decay=k_bn_decay,
            drop=k_drop
        )
        self.W_v = FC(
            input_dims=output_dim,
            units=output_dim,
            activations=get_activation(v_act),
            bn_decay=v_bn_decay,
            drop=v_drop
        )
        self.mapping = FC(
            input_dims=self.output_dim,
            units=self.output_dim,
            activations=get_activation(map_act),
            bn_decay=map_bn_decay,
            drop=map_drop
        )
        self.fusion = FC(
            input_dims=output_dim * 2,
            units=output_dim,
            activations=get_activation(fusion_act),
            bn_decay=fusion_bn_decay,
            drop=fusion_drop
        )

    def forward(self, traffic, nav):
        Q = self.W_q(nav)
        K = self.W_k(traffic)
        V = self.W_v(traffic)

        attn_scores = torch.einsum('btrd,btnd->btrn', Q, K) / torch.sqrt(
            torch.tensor(self.output_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.einsum('btrn,btnd->btrd', attn_weights, V)
        mapped = self.mapping(attended)
        updated_traffic = traffic + mapped.mean(dim=2, keepdim=True)

        return updated_traffic


class STBlock(nn.Module):
    def __init__(self, adj_matrix, input_dim, output_dim, args, top_k=10, bn_decay=0.1, drop=0.1):
        super(STBlock, self).__init__()
        self.spatialAttention = spatialAttention(args.K, args.d, args, drop)
        self.graph_conv = GraphDiffusionConv(adj_matrix, input_dim, output_dim, bn_decay)
        self.key_node_attn = KeyNodeAttention(input_dim, output_dim, args.K, top_k, args, drop)

        self.temporal_attn = TemporalAttention(input_dim, output_dim, args, drop)
        self.time_series_pred = FFTbasedTemporalPeriodExtraction(input_dim, output_dim,
                                                    num_freq=int(args.num_pred / 2) + 1,
                                                    bn_decay=bn_decay)

        # 融合层参数从args获取
        spatial_fusion_bn_decay = 0.1
        spatial_fusion_drop = 0.01
        spatial_fusion_act = 'relu'
        temporal_fusion_bn_decay = 0.1
        temporal_fusion_drop =0.01
        temporal_fusion_act = 'relu'
        final_fusion_bn_decay = 0.1
        final_fusion_drop = 0.01
        final_fusion_act = 'relu'

        self.spatial_fusion = FC(
            input_dims=output_dim * 3,
            units=output_dim,
            activations=get_activation(spatial_fusion_act),
            bn_decay=spatial_fusion_bn_decay,
            drop=spatial_fusion_drop
        )
        self.temporal_fusion = FC(
            input_dims=output_dim * 2,
            units=output_dim,
            activations=get_activation(temporal_fusion_act),
            bn_decay=temporal_fusion_bn_decay,
            drop=temporal_fusion_drop
        )
        self.final_fusion = FC(
            input_dims=output_dim * 2,
            units=output_dim,
            activations=get_activation(final_fusion_act),
            bn_decay=final_fusion_bn_decay,
            drop=final_fusion_drop
        )

        # 分布矫正
        self.output = nn.LayerNorm(output_dim)

    def forward(self, X, STE, is_encoder=True):
        # 空间部分
        spatial_attention = self.spatialAttention(X, STE)
        graph_output = self.graph_conv(X)
        node_output = self.key_node_attn(X, self.graph_conv.norm_adj, STE)
        spatial_output = self.spatial_fusion(torch.cat([spatial_attention,graph_output,node_output], dim=-1))

        # 时间部分
        attn_output = self.temporal_attn(X, STE, mask_future=is_encoder)
        pred_output = self.time_series_pred(X)
        temporal_output = self.temporal_fusion(torch.cat([attn_output,pred_output], dim=-1))

        # 时空融合
        st_output = self.final_fusion(torch.cat([spatial_output, temporal_output], dim=-1))
        st_output = self.output(st_output)
        output = X + st_output

        return output


class TransformAttention(nn.Module):
    def __init__(self, K, d, args, drop=None):
        super(TransformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        q_bn_decay =0.1
        q_drop =0.01
        q_act ='relu'
        k_bn_decay =0.1
        k_drop = 0.01
        k_act ='relu'
        v_bn_decay = 0.1
        v_drop = 0.01
        v_act = 'relu'
        fc_bn_decay = 0.1
        fc_drop = 0.01
        fc_act = 'relu'

        self.FC_q = FC(
            input_dims=D,
            units=D,
            activations=get_activation(q_act),
            bn_decay=q_bn_decay,
            drop=q_drop
        )
        self.FC_k = FC(
            input_dims=D,
            units=D,
            activations=get_activation(k_act),
            bn_decay=k_bn_decay,
            drop=k_drop
        )
        self.FC_v = FC(
            input_dims=D,
            units=D,
            activations=get_activation(v_act),
            bn_decay=v_bn_decay,
            drop=v_drop
        )
        self.FC = FC(
            input_dims=D,
            units=D,
            activations=get_activation(fc_act),
            bn_decay=fc_bn_decay,
            drop=fc_drop
        )

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)

        return X


class STFQET(nn.Module):
    def __init__(self, SE, args, adj_matrix, bn_decay=0.1, drop=0.1):
        super(STFQET, self).__init__()
        self.args = args
        self.register_buffer('SE', SE)

        input_dim = 1
        output_dim = args.d * args.K
        self.fourier_filter = FourierFilter(args)
        traffic_bn_decay = 0.1
        traffic_drop = 0.01
        traffic_act1 = 'relu'
        traffic_act2 = 'none'
        query_bn_decay = 0.1
        query_drop = 0.01
        query_act1 = 'relu'
        query_act2 = 'none'

        self.traffic_linear = FC(
            input_dims=[input_dim, output_dim],
            units=[output_dim, output_dim],
            activations=[get_activation(traffic_act1), get_activation(traffic_act2)],
            bn_decay=traffic_bn_decay,
            drop=traffic_drop
        )
        self.query_linear = FC(
            input_dims=[2, output_dim],
            units=[output_dim, output_dim],
            activations=[get_activation(query_act1), get_activation(query_act2)],
            bn_decay=query_bn_decay,
            drop=query_drop
        )

        self.nav_attention = QueryenhancedAttention(output_dim, args)

        self.st_embedding = STEmbedding(output_dim, args)

        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        if SE.device != adj_matrix.device:
            adj_matrix = adj_matrix.to(SE.device)

        self.encoder_blocks = nn.ModuleList([
            STBlock(adj_matrix, output_dim, output_dim, args,
                    top_k=10, bn_decay=bn_decay)
            for _ in range(args.L)
        ])

        self.transform_attn = TransformAttention(args.K, args.d, args)

        self.decoder_blocks = nn.ModuleList([
            STBlock(adj_matrix, output_dim, output_dim, args,
                    top_k=10, bn_decay=bn_decay)
            for _ in range(args.L)
        ])

        output_bn_decay = 0.1
        output_drop = 0.01
        output_act1 = 'relu'
        output_act2 = 'none'

        self.output_layer = FC(
            input_dims=[output_dim, output_dim],
            units=[output_dim, 1],
            activations=[get_activation(output_act1), get_activation(output_act2)],
            bn_decay=output_bn_decay,
            drop=output_drop
        )

    def forward(self, X, TE, query_data):
        X = torch.unsqueeze(X, -1)

        filtered_X = self.fourier_filter(X)

        traffic_transformed = self.traffic_linear(filtered_X)

        query = query_data.reshape(query_data.shape[0], query_data.shape[1], -1, query_data.shape[4])
        query_transformed = self.query_linear(query)

        updated_traffic = self.nav_attention(traffic_transformed, query_transformed)

        STE = self.st_embedding(self.SE, TE)
        STE_his = STE[:, :self.args.num_his]
        STE_pred = STE[:, self.args.num_his:]

        encoder_output = updated_traffic
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output, STE_his, is_encoder=True)

        transformed = self.transform_attn(encoder_output, STE_his, STE_pred)

        decoder_output = transformed
        for block in self.decoder_blocks:
            decoder_output = block(decoder_output, STE_pred, is_encoder=False)

        output = self.output_layer(decoder_output)
        return output.squeeze(-1)