import os.path

import yaml

def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth+1]), list(cand_tuple[depth + 1: 2 * depth + 1]), cand_tuple[-1]
#
# # res=decode_cand_tuple(top1)
# # print('res is:', res)
#
def build_yaml(info, idx):
    config = {'supernet':{'mlp_ratio': [4.0 for _ in range(16)], 'num_heads': [10 for _ in range(16)], 'layer_num': 16, 'embed_dim': [624]},
              'search_space':{'mlp_ratio': [[3.0, 3.5, 4.0] for _ in range(16)], 'num_heads': [[8, 9, 10] for _ in range(16)], 'layer_num': [14, 15, 16], 'embed_dim': [528, 576, 624]}}
    depth, mlps, m_heads, emb_dim=decode_cand_tuple(info)
    # print('depth:',depth)
    # print('mlps:', mlps)
    # print('m_heads:',m_heads)
    # print('emb_dim:', emb_dim)
    config['supernet']['layer_num'] = depth
    config['supernet']['embed_dim'] = [emb_dim for _ in range(depth)]
    config['supernet']['mlp_ratio'] = mlps
    config['supernet']['num_heads'] = m_heads
    config['search_space']['layer_num'] = [max(depth, 12), max(depth - 1, 12), max(depth - 2, 12)]
    config['search_space']['embed_dim'] = [max(emb_dim, 320), max(emb_dim - 48, 320), max(emb_dim - 96, 320)]

    config['search_space']['mlp_ratio'] = [[max(items, 3), max(items - 0.5, 3), max(items - 1, 3)] for items in mlps]
    config['search_space']['num_heads'] = [[max(m, 5), max(m-1, 5), max(m-2, 5)] for m in m_heads]
    text_info = yaml.safe_dump(config, default_flow_style=False)
    with open('output/'+ 'supernet_iter_{}.yaml'.format(idx+1), 'w') as f:
        f.write(text_info)