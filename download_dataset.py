#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('curl -L -o facebookresearch-mmf-v0.3.1-720-g47ee79b.tar.gz -O "https://github.com/kartikaykaushik14/HatefulMemes/blob/main/Modified%20Libraries/facebookresearch-mmf-v0.3.1-720-g47ee79b.tar.gz?raw=true"')
get_ipython().system('pip install facebookresearch-mmf-v0.3.1-720-g47ee79b.tar.gz')


# In[5]:


get_ipython().system('pip install --upgrade --no-cache-dir gdown')
get_ipython().system('gdown --id 1VDhexWsU0pFIZ8nDKg7nC6CSOsR9ExeA')
get_ipython().system('mmf_convert_hm --zip_file ./hateful_memes.zip --password "password" --bypass_checksum=1 --mmf_data_folder ./')