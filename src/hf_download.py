# selective download of tran et al. shifted splits from huggingface
# repo: https://huggingface.co/datasets/nntvu/MalNet-Tiny-Features
# we skip llm embeddings (huge), pull graph structures and labels only

import os


HF_REPO = "nntvu/MalNet-Tiny-Features"


def list_repo_files():
    # quick inspection of what files exist in the repo, helps figure out the layout
    # wrap in list() since recent huggingface_hub returns an iterable, not a list
    from huggingface_hub import HfApi
    api = HfApi()
    return list(api.list_repo_files(HF_REPO, repo_type="dataset"))


def download_common_split(local_dir="data/hf_malnet/", variant="none+none"):
    # only pull a single feature variant subdir, not the whole TB repo
    # variant="none+none" means structure + labels only, no metadata or llm embeddings
    # other variants exist (zero+base, prune+base, zero+llm, etc.) but are huge
    # ref: https://huggingface.co/docs/huggingface_hub/guides/download
    from huggingface_hub import snapshot_download

    # fnmatch * matches across path separators so /* is enough for nested files too
    # raw/ does not exist on this hf repo, only processed/, so we skip it
    allow = [
        f"MalNetTinyFeaturesCommon/processed/{variant}/*",
    ]

    path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=local_dir,
    )
    return path


def get_split_root(local_dir="data/hf_malnet/", variant="none+none"):
    # returns the path to the specific variant subdir we downloaded, useful for shift loader
    candidate = os.path.join(local_dir, "MalNetTinyFeaturesCommon", "processed", variant)
    if os.path.exists(candidate):
        return candidate
    # fall back to whatever exists
    common = os.path.join(local_dir, "MalNetTinyFeaturesCommon")
    if os.path.exists(common):
        return common
    return local_dir
