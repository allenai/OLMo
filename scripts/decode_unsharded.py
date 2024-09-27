import sys
import torch
import transformers
from olmo.config import TrainConfig, InfgramConfig
from olmo.model import OLMo
from olmo.checkpoint import load_state_dict
from olmo.util import clean_opt
from olmo.exceptions import OLMoCliError
from infini_gram import InfiniGramEngine

def main(cfg: TrainConfig) -> None:
    infgram_config = InfgramConfig(
        index_dir='/net/nfs.cirrascale/allennlp/jiachengl/hb-wolf/index/v5_dolma-v1_7-wiki_olmo',
        mode='debug', separate_wte=True, support=1,
    )
    infinigram_engine = InfiniGramEngine(
        cfg=infgram_config,
        max_batch_size_per_device=1024,
        max_seq_len=cfg.model.max_sequence_length,
        local_rank=0,
        global_rank=0,
        local_world_size=1,
        world_size=1,
    )

    cfg.infgram = infgram_config
    cfg.model.init_device = "cpu"

    olmo_model = OLMo(cfg.model, separate_infgram_wte=cfg.infgram.separate_wte if cfg.infgram is not None else False)
    # load_path = './ckpt_unsharded/v2.7_v2.5_vera_no-infgram_step11234'
    load_path = './ckpt_unsharded/v2.5_v2.4_shane-fix_step409000'
    state_dict_to_load = load_state_dict(
        load_path, "model.pt", local_cache=None, map_location="cpu"
    )
    olmo_model.load_state_dict(state_dict_to_load, strict=True)

    MODEL_NAME = './ckpt_transformers/v2.5_v2.4_shane-fix_step409000'
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    # message = ["Barack Obama was born in"]
    message = ["The best way to learn is to teach others what you"]
    inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    # infgram_ntd = infinigram_engine.get_infgram_ntd(input_idss=input_ids, method=2)['infgram_ntd']
    # print(f'input_ids = {inputs["input_ids"]}')
    # print(f'infgram_ntd = {infgram_ntd}')
    # logits = olmo_model(
    #     input_ids=inputs['input_ids'],
    #     attention_mask=inputs['attention_mask'],
    #     infgram_ntd=infgram_ntd,
    # ).logits # (B, L, V)
    # logits = logits[0, -1] # (V,)
    # print(logits.topk(5))
    # exit()

    for i in range(50):
        infgram_ntd = infinigram_engine.get_infgram_ntd(input_idss=input_ids, method=2)['infgram_ntd']
        logits = olmo_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            infgram_ntd=infgram_ntd,
        ).logits[:, -1] # (B, V)
        next_token_id = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
    message = tokenizer.decode(input_ids[0])
    print(message)

if __name__ == "__main__":
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
