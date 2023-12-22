# Visualizing Expert Firing Frequencies in Mixtral MoE

Recently, Mistral AI's Mixture-of-Experts model ([Mixtral MoE](https://mistral.ai/news/mixtral-of-experts/)) shows impressive performance, comparable to a 70B model, despite only requiring forward pass capacity of a 13B model. The model dynamically decides which expert to use for each token, where an expert is an FFN or an MLP layer in a traditional transformer model. Specifically, there are 8 different expert MLPs at each layer out of which 2 are picked by a module called router for a given token. Given that the model can now choose which MLP layers to use for each token, unlike attention modules, it is reasonable to believe that the experts are, well, experts on different topics. This project attempts to visualize whether this actually happens. Check it out in action at http://mixtral-moe-vis-d726c4a10ef5.herokuapp.com.

If you find our work valuable, please cite:

```
@online{mixtral-vis-moe,
  author    = {Ajinkya Tejankar and Hamed Pirsiavash},
  title     = {Visualizing Expert Firing Frequencies in Mixtral MoE},
  url       = {https://github.com/ajtejankar/mixtral-vis-moe},
  year      = {2023},
  month     = {Dec}
}
```
