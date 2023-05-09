# OLMo evaluation

## Downstream task evaluation during pretraining

This contains basic plumbing to hook OLMo into catwalk, using code currently 
available at https://github.com/OyvindTafjord/catwalk. Best used by running
beaker-gantry from within the catwalk repo, pointing to an appropriate beaker image,
such as [`oyvindt/OLMoEvalV4`](https://beaker.org/im/01GZZ48Q0CNJ1QD20QCH891306/details)
(typically pick the highest version number).

See the Jupyter Notebook(s) in this directory for examples of launching Beaker
experiments.

See [the "OLMo Evaluation during pre-training" document](https://docs.google.com/document/d/1HahVawRR2Nf_J_B5Adsxierp4HK01tV8NR9o6NFUgMo/edit?usp=sharing) 
for more information.

For updating the beaker image, here's one approach which pulls an existing base image, and
does some mumbo-jumbo to make it build:

```commandline
$ beaker image pull petew/olmo-torch2-gantry olmo-torch2-gantry
$ DOCKER_BUILDKIT=0 docker build --pull=false -f docker/Dockerfile.catwalk -t olmo-eval:20230508.4 .
$ beaker image create -n OLMoEvalV4 olmo-eval:20230508.4
```

