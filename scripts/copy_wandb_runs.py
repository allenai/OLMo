import wandb
from wandb.wandb_run import Run

ENTITY = "ai2-llm"
DST_PROJECT = "OLMo-7B"

runs_to_copy = [
    ###########################################################################################
    ######################################### OLMo-7B #########################################
    ###########################################################################################
    #  ("ai2-llm/olmo-medium/runs/wvc30anm", "OLMo-7B-run-001", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/uhy9bs35", "OLMo-7B-run-002", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/l6v218f4", "OLMo-7B-run-003", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/8fioq3qx", "OLMo-7B-run-004", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/mk9kaqh0", "OLMo-7B-run-005", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/49i87wpn", "OLMo-7B-run-006", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/0j2eqydw", "OLMo-7B-run-007", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/5wkmhkqh", "OLMo-7B-run-008", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/hrshlkzq", "OLMo-7B-run-009", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/eysi0t0y", "OLMo-7B-run-010", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/7gomworq", "OLMo-7B-run-011", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/lyij2l8m", "OLMo-7B-run-012", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/99euueq4", "OLMo-7B-run-013", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/fcn5q3zw", "OLMo-7B-run-014", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/j18wauyq", "OLMo-7B-run-015", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/jtfwv96r", "OLMo-7B-run-016", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/yuc5kl7s", "OLMo-7B-run-017", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/25urleov", "OLMo-7B-run-018", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/obde4w9j", "OLMo-7B-run-019", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/eaqax5ns", "OLMo-7B-run-020", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/cojbrc1o", "OLMo-7B-run-021", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/4xel5n7e", "OLMo-7B-run-022", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/jcs4c32w", "OLMo-7B-run-023", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/x55jyv7k", "OLMo-7B-run-024", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/yv7lgx0i", "OLMo-7B-run-025", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/11uf7gsv", "OLMo-7B-run-026", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/lds6zcog", "OLMo-7B-run-027", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/ho7jy4ey", "OLMo-7B-run-028", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/87shig0a", "OLMo-7B-run-029", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/x6zdcp5j", "OLMo-7B-run-030", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/olocmvn0", "OLMo-7B-run-031", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/xtruaap8", "OLMo-7B-run-032", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/2l070ogq", "OLMo-7B-run-033", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/uy2ydw12", "OLMo-7B-run-034", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/x23ciyv9", "OLMo-7B-run-035", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/67i5mdg0", "OLMo-7B-run-036", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/wrv46m83", "OLMo-7B-run-037", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/wd2gxrza", "OLMo-7B-run-038", "OLMo-7B"),
    #  ("ai2-llm/olmo-medium/runs/z4z0x4m9", "OLMo-7B-run-039", "OLMo-7B"),
    ###########################################################################################
    ####################### OLMo-7B fine-tuned on a mix of Tulu + Dolma #######################
    ###########################################################################################
    #  ("ai2-llm/olmo-medium/runs/p067ktg9", "OLMo-7B-Tulu", "OLMo-7B-Tulu"),
    ###########################################################################################
    ##################################### OLMo-7B-Twin-2T #####################################
    ###########################################################################################
    #  ("ai2-llm/olmo-medium/runs/fi03r8h0", "OLMo-7B-Twin-2T-run-001", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/2fi6zuqd", "OLMo-7B-Twin-2T-run-002", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/j8qk3cgd", "OLMo-7B-Twin-2T-run-003", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/gd4pltei", "OLMo-7B-Twin-2T-run-004", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/vpxr4bil", "OLMo-7B-Twin-2T-run-005", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/vgkz4o76", "OLMo-7B-Twin-2T-run-006", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/lp1ynh47", "OLMo-7B-Twin-2T-run-007", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/03rx6g79", "OLMo-7B-Twin-2T-run-008", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/aznf5iwj", "OLMo-7B-Twin-2T-run-009", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/nzw0h387", "OLMo-7B-Twin-2T-run-010", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/v6je6zon", "OLMo-7B-Twin-2T-run-011", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/im54vfs8", "OLMo-7B-Twin-2T-run-012", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/m9j3x5o0", "OLMo-7B-Twin-2T-run-013", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/rto0vcbk", "OLMo-7B-Twin-2T-run-014", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/7qe3jywj", "OLMo-7B-Twin-2T-run-015", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/71r8xird", "OLMo-7B-Twin-2T-run-016", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/kya6t318", "OLMo-7B-Twin-2T-run-017", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/3rvuwvew", "OLMo-7B-Twin-2T-run-018", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/cmash3bz", "OLMo-7B-Twin-2T-run-019", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/qs7w6w53", "OLMo-7B-Twin-2T-run-020", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/z1gcqs1y", "OLMo-7B-Twin-2T-run-021", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/lualc6sf", "OLMo-7B-Twin-2T-run-022", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/s63r1dze", "OLMo-7B-Twin-2T-run-023", "OLMo-7B-Twin-2T"),
    #  ("ai2-llm/olmo-medium/runs/n761ckim", "OLMo-7B-Twin-2T-run-024", "OLMo-7B-Twin-2T"),
    ###########################################################################################
    ######################################### OLMo-1B #########################################
    ###########################################################################################
    #  ("ai2-llm/olmo-small/runs/w1r5xfzt", "OLMo-1B-run-001", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/s7wptaol", "OLMo-1B-run-002", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/sw58clgr", "OLMo-1B-run-003", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/age4ucpn", "OLMo-1B-run-004", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/9lhyy6ec", "OLMo-1B-run-005", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/e72w3guf", "OLMo-1B-run-006", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/czmq3tph", "OLMo-1B-run-007", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/xcki6amz", "OLMo-1B-run-008", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/9o4tqzkc", "OLMo-1B-run-009", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/scjaj9rj", "OLMo-1B-run-010", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/ecm3b6jc", "OLMo-1B-run-011", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/tm06cx1o", "OLMo-1B-run-012", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/hv91c1yz", "OLMo-1B-run-013", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/i27fd8hx", "OLMo-1B-run-014", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/nj3eug16", "OLMo-1B-run-015", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/qkgvoqxh", "OLMo-1B-run-016", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/vohm89rs", "OLMo-1B-run-017", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/bdal15q1", "OLMo-1B-run-018", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/k7gf8upq", "OLMo-1B-run-019", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/9mx2iel7", "OLMo-1B-run-020", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/epwms9w9", "OLMo-1B-run-021", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/8qy0al8a", "OLMo-1B-run-022", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/e3hcu37o", "OLMo-1B-run-023", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/5gqrwqg1", "OLMo-1B-run-024", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/mbho3mal", "OLMo-1B-run-025", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/3v73eans", "OLMo-1B-run-026", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/7l54afq9", "OLMo-1B-run-027", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/mif67a9e", "OLMo-1B-run-028", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/9frhziu4", "OLMo-1B-run-029", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/46zc5fly", "OLMo-1B-run-030", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/n9ya5dg7", "OLMo-1B-run-031", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/sezmr7ds", "OLMo-1B-run-032", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/185jyoim", "OLMo-1B-run-033", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/4ryfkyyz", "OLMo-1B-run-034", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/njlk53cc", "OLMo-1B-run-035", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/gqbsxin2", "OLMo-1B-run-036", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/berj88t7", "OLMo-1B-run-037", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/py49d2az", "OLMo-1B-run-038", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/39zrvyeq", "OLMo-1B-run-039", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/oupb6jak", "OLMo-1B-run-040", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/w4ele4r3", "OLMo-1B-run-041", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/qt3d0ypt", "OLMo-1B-run-042", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/h8d87477", "OLMo-1B-run-043", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/h9g4p1cw", "OLMo-1B-run-044", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/3fii7eec", "OLMo-1B-run-045", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/jy5po36u", "OLMo-1B-run-046", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/s73qq2ny", "OLMo-1B-run-047", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/rs1scdrz", "OLMo-1B-run-048", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/dcd1wqlw", "OLMo-1B-run-049", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/3xqkbrvw", "OLMo-1B-run-050", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/xgc3lo3d", "OLMo-1B-run-051", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/yifb3rvs", "OLMo-1B-run-052", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/q1qhhvxg", "OLMo-1B-run-053", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/mkunaie6", "OLMo-1B-run-054", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/rg0wtuij", "OLMo-1B-run-055", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/xbvqtb2c", "OLMo-1B-run-056", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/mvuu3vcl", "OLMo-1B-run-057", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/bvix71p0", "OLMo-1B-run-058", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/44to2rqh", "OLMo-1B-run-059", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/8fl06671", "OLMo-1B-run-060", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/b3zzyyc7", "OLMo-1B-run-061", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/4k49us4j", "OLMo-1B-run-062", "OLMo-1B"),
    #  ("ai2-llm/olmo-small/runs/g4g72enr", "OLMo-1B-run-063", "OLMo-1B"),
    ###########################################################################################
    ##################################### OLMo-1.7-7B #########################################
    ###########################################################################################
    ("ai2-llm/olmo-annealing/runs/yu3ctnnk", "OLMo-1.7-7B-anneal-50B", "OLMo-1.7-7B-anneal"),
    #  ("ai2-llm/olmo-medium/runs/0o2xzqba", "OLMo-1.7-7B-run-025", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/f1env9wp", "OLMo-1.7-7B-run-024", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/orwrnhrm", "OLMo-1.7-7B-run-023", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/g1q44n0t", "OLMo-1.7-7B-run-022", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/a3uhfztd", "OLMo-1.7-7B-run-021", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/a6wf5h5x", "OLMo-1.7-7B-run-020", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/y88c938f", "OLMo-1.7-7B-run-019", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/0rdfxd6d", "OLMo-1.7-7B-run-018", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/vis76vmr", "OLMo-1.7-7B-run-017", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/6qvoqf3c", "OLMo-1.7-7B-run-016", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/hdzw14gs", "OLMo-1.7-7B-run-015", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/8hnxbu0t", "OLMo-1.7-7B-run-014", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/pj8uxkvy", "OLMo-1.7-7B-run-013", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/o82kqvjh", "OLMo-1.7-7B-run-012", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/tv40o5gq", "OLMo-1.7-7B-run-011", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/f42888qx", "OLMo-1.7-7B-run-010", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/jw164eti", "OLMo-1.7-7B-run-009", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/adpt1srg", "OLMo-1.7-7B-run-008", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/rtuyob91", "OLMo-1.7-7B-run-007", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/gckmq5es", "OLMo-1.7-7B-run-006", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/2qni92dc", "OLMo-1.7-7B-run-005", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/s84zsd99", "OLMo-1.7-7B-run-004", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/fgzk495l", "OLMo-1.7-7B-run-003", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/ie0bx486", "OLMo-1.7-7B-run-002", "OLMo-1.7-7B"),
    #  ("ai2-llm/olmo-medium/runs/geox6yo7", "OLMo-1.7-7B-run-001", "OLMo-1.7-7B"),
]

# Set your API key
wandb.login()

# Initialize the wandb API
api = wandb.Api()

# Iterate through the runs and copy them to the destination project
for run_path, new_run_name, new_run_group in sorted(runs_to_copy, key=lambda x: x[1]):
    run = api.run(run_path)

    print(f"Copying run '{run_path}' to '{DST_PROJECT}/{new_run_name}'...")

    # Get the run history and files
    history = run.scan_history()

    # Create a new run in the destination project
    new_run = wandb.init(
        project=DST_PROJECT,
        entity=ENTITY,
        config=run.config,
        name=new_run_name,
        resume="allow",
        group=new_run_group,
        settings=wandb.Settings(_disable_stats=True),
    )
    assert isinstance(new_run, Run)

    # Log the history to the new run
    for data in history:
        step = data.pop("_step")
        new_run.log(data, step=step)

    # Finish the new run
    new_run.finish()
