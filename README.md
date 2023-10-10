# Code for "We Are What We Repeatedly Do: Inducing and Deploying Habitual Schemas in Persona-Based Responses"

Benjamin Kane and Lenhart Schubert. [We Are What We Repeatedly Do: Inducing and Deploying Habitual Schemas in Persona-Based Responses](TODO), EMNLP 2023 (to appear).


## Dependencies

See requirements.txt. A valid openai API key is also required in `_keys/openai.txt`.


## Data

The `data`` directory contains our expansion of the PersonaChat test set containing schemas and responses generated by each of the methods described in our paper.


## How to run

`cd src`

### Schema generation

To estimate costs:

`python3 generate_schemas.py --dataset "personachat" --percent 100`

To generate:

`python3 generate_schemas.py --dataset "personachat" --percent 100 --output-filename "../data/personachat-schemas.json" --checkpoint-iter 50 --generate`

### Response generation

To estimate costs:

`python3 generate_response.py --dataset "../data/personachat-schemas.json" --percent 100`

To generate:

`python3 generate_response.py --dataset "../data/personachat-schemas.json" --percent 100 --output-filename "../data/personachat-schemas-responses.json" --checkpoint-iter 50 --generate`


## Citation

If you find our response generation methods useful for your research, please cite our paper:

```BibTex
@inproceedings{Kane2023Habitual,
  author    = {Benjamin Kane and Lenhart Schubert},
  title     = {We Are What We Repeatedly Do: Inducing and Deploying Habitual Schemas in Persona-Based Responses},
  booktitle = {EMNLP},
  year      = {2023},
  url       = {TODO},
}
```