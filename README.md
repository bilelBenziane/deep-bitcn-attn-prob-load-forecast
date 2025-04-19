# An Ensemble Framework for Probabilistic Short-Term Load Forecasting Based on BiTCN and Deep Attention Networks

_An Ensemble Framework for Probabilistic Short-Term Load Forecasting Based on BiTCN and Deep Attention Networks_ is a repository containing code for a new forecasting method for power load forecasting. This architecture is trained and tested against state-of-the-art methods using the code provided by the paper [Parameter Efficient Deep Probabilistic Forecasting (PEDPF)](https://www.sciencedirect.com/science/article/pii/S0169207021001850). For more details, see [our paper](https://www.techrxiv.org/doi/full/10.36227/techrxiv.174235483.35557269).

<img src="figures/bitcn_deep_att_skip.jpg" alt="Experimental Setup" width="80%" />

---

### 📂 Added Files

#### `algorithms/`
- `bitcn_att_skip.py` — BiTCN with skip connections and deep attention  
- `bitcn_att_no_skip.py` — BiTCN with deep attention but no skip connections  

#### Root directory
- `train_test_electricity.py` — Main training and evaluation script for the electricity dataset  
- `paper figure generator.ipynb` — Notebook to generate figures from paper results  

---

### 📈 Reproduce Paper's Experiments
1. **Prepare a conda environment**:

   ```bash
   conda create -n load_forecast_env python=3.10
   conda activate load_forecast_env
   pip install numpy==1.24.4
   pip install pyparsing scipy jinja2 packaging psutil pyyaml tornado joblib pynisher pyrfr
   pip install scikit-learn==1.3.2
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
   pip install jupyterlab
   pip install matplotlib
   pip install pandas==1.5.3

   ```

2. clone the project
   ```bash
   git clone https://github.com/bilelBenziane/deep-bitcn-attn-prob-load-forecast.git
   cd deep-bitcn-attn-prob-load-forecast
   ```
4. Download the required dataset:
   - [UCI Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
   - Extract `LD2011_2014.txt` to the folder `data/uci_electricity/` (create the folder if needed)

5. Run the main experiment script:

   ```bash
   python train_test_electricity.py
   ```
6. Then you can refer to `paper figure generator.ipynb` to generate the figures of the paper.



### Reference ###
[Bilel Benziane](mailto:bilel.benziane@isen-ouest.yncrea.fr), Benoit Lardeux, Maher Jridi, Ayoub Mcharek. [An Ensemble Framework for Probabilistic Short-Term Load Forecasting Based on BiTCN and Deep Attention Networks](https://www.techrxiv.org/doi/full/10.36227/techrxiv.174235483.35557269). Submitted as a journal paper to [InternaIEEE Transactions on Power Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=59).

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/pedpf/blob/master/LICENSE).

### Acknowledgements ###
This project was a contribution of a new forecast architecture tested on the experimental protocol developed by [Airlab Amsterdam](https://icai.ai/airlab/).
