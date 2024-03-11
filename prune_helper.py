import torch

def prune_lowindex(new_model, old_model):
    # print(new_model.modules)
    # print(old_model.modules)
    for [a, b] in zip(new_model.modules(), old_model.modules()):
    #     # print('what is going on?')
    #     print('new model:', a)
    #     print('old model:', b)
        if a._get_name() == 'PatchembedSuper' and b._get_name() == 'PatchembedSuper':
            print(a)
            print(b)
            a.proj.weight.data = b.proj.weight.data[:a.super_embed_dim, ...].clone()
            a.proj.bias.data = b.proj.bias.data[:a.super_embed_dim, ...].clone()
            a.sampled_weight = b.proj.weight.data[:a.super_embed_dim, ...].clone()
            a.sampled_bias = b.proj.bias.data[:a.super_embed_dim, ...].clone()
            print('\n')

        if a._get_name() == 'LayerNormSuper' and b._get_name() == 'LayerNormSuper':
            print(a)
            print(b)
            a.weight.data = b.weight.data[:a.super_embed_dim].clone()
            b.bias.data = b.bias.data[:a.super_embed_dim].clone()
            a.samples['weight'] = b.weight.data[:a.super_embed_dim].clone()
            a.samples['bias'] = b.bias.data[:a.super_embed_dim].clone()
            print('\n')

        if a._get_name() == 'LinearSuper' and b._get_name() == 'LinearSuper':
            print(a)
            print(b)
            # print(b)
            a.weight.data = b.weight.data[:a.super_out_dim, :a.super_in_dim].clone()

            a.samples['weight'] = b.weight.data[:a.super_out_dim, :a.super_in_dim].clone()
            a.samples['bias'] = b.bias
            if a.bias is not None:
                a.bias.data = b.bias.data[:a.super_out_dim].clone()
                a.samples['bias'].data = b.bias.data[:a.super_out_dim].clone()
            print('\n')

        if a._get_name() == 'qkv_super' and b._get_name() == 'qkv_super':
            print(a)
            print(b)
            # print(b)
            w_a = b.weight[:, :a.super_in_dim]
            a.weight.data = torch.cat([w_a[i:a.super_out_dim:3, :] for i in range(3)], dim=0)
            a.samples['weight'] = torch.cat([w_a[i:a.super_out_dim:3, :] for i in range(3)], dim=0)
            a.samples['bias'] = b.bias
            if a.bias is not None:
                a.bias.data = b.bias.data[:a.super_out_dim].clone()
                a.samples['bias'] = b.bias[:a.super_out_dim].clone()
            print('\n')

        if a._get_name() == 'RelativePosition2D_super' and b._get_name() == 'RelativePosition2D_super':
            print(a)
            print(b)
            a.embeddings_table_h.data = b.embeddings_table_h.data[:, :a.num_units].clone()
            a.embeddings_table_v.data = b.embeddings_table_v.data[:, :a.num_units].clone()
            a.sample_embeddings_table_h = b.embeddings_table_h[:, :a.num_units]
            a.sample_embeddings_table_v = b.embeddings_table_v[:, :a.num_units]
            print('\n')

        else:
            # if a._get_name() == 'TransformerEncoderLayer' and b._get_name() == 'TransformerEncoderLayer':
            # print(a)
            # print(b)
            continue

        return new_model


def new_prune(model, checkpoint):
    new_state_dict = model.state_dict()
    for k, v in checkpoint['model'].items():
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k
        if name in new_state_dict.keys():
            # print(name)

            if name.startswith('patch_embed_super'):
                print(name)
                new_state_dict[name] = v[:model.patch_embed_super.super_embed_dim, ...]

            if name.startswith('blocks') and name.split('.')[3] == 'qkv' and name.split('.')[2] == 'attn':
                print(name)
                this_block_id = int(name.split('.')[1])
                this_block = model.blocks[this_block_id].attn.qkv
                in_dim = this_block.super_in_dim
                out_dim = this_block.super_out_dim
                if name.split('.')[4] == 'weight':
                    tmp = v[:, :in_dim]
                    new_state_dict[name] = torch.cat([tmp[i:out_dim:3, :] for i in range(3)], dim=0)
                if name.split('.')[4] == 'bias':
                    new_state_dict[name] = v[:out_dim]

            if name.startswith('blocks') and name.split('.')[3] == 'proj' and name.split('.')[2] == 'attn':
                print(name)
                this_block_id = int(name.split('.')[1])
                this_block = model.blocks[this_block_id].attn.proj
                in_dim = this_block.super_in_dim
                out_dim = this_block.super_out_dim
                if name.split('.')[4] == 'weight':
                    tmp = v[:, :in_dim]
                    new_state_dict[name] = tmp[:out_dim, :]
                if name.split('.')[4] == 'bias':
                    new_state_dict[name] = v[:out_dim]

            if name.startswith('blocks') and name.split('.')[2] == 'attn_layer_norm':
                print(name)
                this_block_id = int(name.split('.')[1])
                this_block = model.blocks[this_block_id].attn_layer_norm
                emb_dim = this_block.super_embed_dim
                new_state_dict[name] = v[:emb_dim]

            if name.startswith('blocks') and name.split('.')[2] == 'ffn_layer_norm':
                print(name)
                this_block_id = int(name.split('.')[1])
                this_block = model.blocks[this_block_id].ffn_layer_norm
                ffn_emb_dim = this_block.super_embed_dim
                new_state_dict[name] = v[:ffn_emb_dim]

            if name.startswith('blocks') and name.split('.')[2] == 'fc1':
                print(name)
                this_block_id = int(name.split('.')[1])
                this_block = model.blocks[this_block_id].fc1
                in_dim = this_block.super_in_dim
                out_dim = this_block.super_out_dim
                if name.split('.')[3] == 'weight':
                    tmp = v[:, :in_dim]
                    new_state_dict[name] = tmp[:out_dim, :]
                if name.split('.')[3] == 'bias':
                    new_state_dict[name] = v[:out_dim]

            if name.startswith('blocks') and name.split('.')[2] == 'fc2':
                print(name)
                this_block_id = int(name.split('.')[1])
                this_block = model.blocks[this_block_id].fc2
                in_dim = this_block.super_in_dim
                out_dim = this_block.super_out_dim
                if name.split('.')[3] == 'weight':
                    tmp = v[:, :in_dim]
                    new_state_dict[name] = tmp[:out_dim, :]
                if name.split('.')[3] == 'bias':
                    new_state_dict[name] = v[:out_dim]

            if name.startswith('norm'):
                print(name)
                this_block = model.norm
                emb_dim = this_block.super_embed_dim
                new_state_dict[name] = v[:emb_dim]

            if name.startswith('head'):
                print(name)
                this_block = model.head
                in_dim = this_block.super_in_dim
                out_dim = this_block.super_out_dim
                if name.split('.')[1] == 'weight':
                    tmp = v[:, :in_dim]
                    new_state_dict[name] = tmp[:out_dim, :]
                if name.split('.')[1] == 'bias':
                    new_state_dict[name] = v[:out_dim]
    model.load_state_dict(new_state_dict)
    return model