# Content Agnostic Moderation for Stance Neutral Recommendation
### under review

## setup
* create a virtual environment with python version 3.10
* install the requirements using `pip install -r requirements.txt`

## Run simulation
* Change config file in `config.yaml`, with different combination of recommender, moderator, and dataset
* run `python main.py`
* observe results in the `results` folder

## Simulation Framework Details
Our simulation framework offers several enhancements over existing models to improve realism and experimental specificity:

1. Dynamic Recommendations: Oracle recommender operates over time rather than in a single calculation step
2. Cold-Start Handling: Each user initially sees 50 items; cold items shown to 10 random users
3. Multi-Item Sessions: Using k > 1 recommendations per session for realistic interactions
4. Controlled Stance Distribution: Systematic sampling of items from different stances
5. Single-Topic Focus: All items relate to one topic to isolate stance dynamics
6. Flexible User Models: Both static and dynamic preference models to separate recommender bias from user self-amplification
