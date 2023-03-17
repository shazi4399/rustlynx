# README #

* In rustlynx/cfg, there are two .toml files for Party0 and Party1. These are settings that describe the network, threads, number system, and the type of ml model to run. All of these presets should work for testing on your local machine, but you can tweak the threads or portranges if you want. For AWS testing, you'll just need to change the IP addresses.

* In rustlynx/cfg/ml/naivebayes/, there are two more settings files inference0 and inferencd1. For any test, you need to specify a path to secret share files containing the dictionary, example, log probabilities and class priors. There should be a separate one each party. 

* The fileIO system I wrote is very basic so make sure to do the following: 
    - Files containing Z2 shares have to be in hex and prefixed with "0x".
    - Files containing Zq  shares have to be in base 10
    - A 1D vector needs to be presented in a .csv with one entry per line
    - A 2D vector needs to be presented in a .csv with all entries per line separated by a comma with no spaces
    - Make sure the other fields in inference0.toml and inference1.toml match the dimensions of the files

### To execute the code in Windows powershell ###

navigate to the root directory rustlynx and type the command "launch.PS1"

### To execute the code using a different terminal ###

* open two terminals and navigate to rustlynx/ with each
* in one of the terminals, enter `cargo build --release`
* in the first, enter `cargo run cfg=cfg/Party0_learning.toml`
* in the second, enter  `cargo run cfg=cfg/Party1_learning.toml`  

### What is this repository for? ###

* MPC Overview
* Version

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests --> cargo test
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact