
# Homework 2
## Yanrong Wu 1801212952

# Problem 1

## 1.Closed form function

Q: Implement a function closed_form_1 that computes this closed form solution given the features 𝐗, labels Y (using Python or Matlab).


```python
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt

climate_change_1 = pd.read_csv('climate_change_1.csv')
climate_change_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>MEI</th>
      <th>CO2</th>
      <th>CH4</th>
      <th>N2O</th>
      <th>CFC-11</th>
      <th>CFC-12</th>
      <th>TSI</th>
      <th>Aerosols</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1983</td>
      <td>5</td>
      <td>2.556</td>
      <td>345.96</td>
      <td>1638.59</td>
      <td>303.677</td>
      <td>191.324</td>
      <td>350.113</td>
      <td>1366.1024</td>
      <td>0.0863</td>
      <td>0.109</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1983</td>
      <td>6</td>
      <td>2.167</td>
      <td>345.52</td>
      <td>1633.71</td>
      <td>303.746</td>
      <td>192.057</td>
      <td>351.848</td>
      <td>1366.1208</td>
      <td>0.0794</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1983</td>
      <td>7</td>
      <td>1.741</td>
      <td>344.15</td>
      <td>1633.22</td>
      <td>303.795</td>
      <td>192.818</td>
      <td>353.725</td>
      <td>1366.2850</td>
      <td>0.0731</td>
      <td>0.137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1983</td>
      <td>8</td>
      <td>1.130</td>
      <td>342.25</td>
      <td>1631.35</td>
      <td>303.839</td>
      <td>193.602</td>
      <td>355.633</td>
      <td>1366.4202</td>
      <td>0.0673</td>
      <td>0.176</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1983</td>
      <td>9</td>
      <td>0.428</td>
      <td>340.17</td>
      <td>1648.40</td>
      <td>303.901</td>
      <td>194.392</td>
      <td>357.465</td>
      <td>1366.2335</td>
      <td>0.0619</td>
      <td>0.149</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1983</td>
      <td>10</td>
      <td>0.002</td>
      <td>340.30</td>
      <td>1663.79</td>
      <td>303.970</td>
      <td>195.171</td>
      <td>359.174</td>
      <td>1366.0589</td>
      <td>0.0569</td>
      <td>0.093</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1983</td>
      <td>11</td>
      <td>-0.176</td>
      <td>341.53</td>
      <td>1658.23</td>
      <td>304.032</td>
      <td>195.921</td>
      <td>360.758</td>
      <td>1366.1072</td>
      <td>0.0524</td>
      <td>0.232</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1983</td>
      <td>12</td>
      <td>-0.176</td>
      <td>343.07</td>
      <td>1654.31</td>
      <td>304.082</td>
      <td>196.609</td>
      <td>362.174</td>
      <td>1366.0607</td>
      <td>0.0486</td>
      <td>0.078</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1984</td>
      <td>1</td>
      <td>-0.339</td>
      <td>344.05</td>
      <td>1658.98</td>
      <td>304.130</td>
      <td>197.219</td>
      <td>363.359</td>
      <td>1365.4261</td>
      <td>0.0451</td>
      <td>0.089</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1984</td>
      <td>2</td>
      <td>-0.565</td>
      <td>344.77</td>
      <td>1656.48</td>
      <td>304.194</td>
      <td>197.759</td>
      <td>364.296</td>
      <td>1365.6618</td>
      <td>0.0416</td>
      <td>0.013</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1984</td>
      <td>3</td>
      <td>0.131</td>
      <td>345.46</td>
      <td>1655.77</td>
      <td>304.285</td>
      <td>198.249</td>
      <td>365.044</td>
      <td>1366.1697</td>
      <td>0.0383</td>
      <td>0.049</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1984</td>
      <td>4</td>
      <td>0.331</td>
      <td>346.77</td>
      <td>1657.68</td>
      <td>304.389</td>
      <td>198.723</td>
      <td>365.692</td>
      <td>1365.5660</td>
      <td>0.0352</td>
      <td>-0.019</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>5</td>
      <td>0.121</td>
      <td>347.55</td>
      <td>1649.33</td>
      <td>304.489</td>
      <td>199.233</td>
      <td>366.317</td>
      <td>1365.7783</td>
      <td>0.0324</td>
      <td>0.065</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>6</td>
      <td>-0.142</td>
      <td>346.98</td>
      <td>1634.13</td>
      <td>304.593</td>
      <td>199.858</td>
      <td>367.029</td>
      <td>1366.0956</td>
      <td>0.0302</td>
      <td>-0.016</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>7</td>
      <td>-0.138</td>
      <td>345.55</td>
      <td>1629.89</td>
      <td>304.722</td>
      <td>200.671</td>
      <td>367.893</td>
      <td>1366.1145</td>
      <td>0.0282</td>
      <td>-0.024</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1984</td>
      <td>8</td>
      <td>-0.179</td>
      <td>343.20</td>
      <td>1643.67</td>
      <td>304.871</td>
      <td>201.710</td>
      <td>368.843</td>
      <td>1365.9781</td>
      <td>0.0260</td>
      <td>0.034</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1984</td>
      <td>9</td>
      <td>-0.082</td>
      <td>341.35</td>
      <td>1663.60</td>
      <td>305.021</td>
      <td>202.972</td>
      <td>369.800</td>
      <td>1365.8669</td>
      <td>0.0239</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1984</td>
      <td>10</td>
      <td>0.016</td>
      <td>341.68</td>
      <td>1674.65</td>
      <td>305.158</td>
      <td>204.407</td>
      <td>370.782</td>
      <td>1365.7869</td>
      <td>0.0220</td>
      <td>-0.035</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1984</td>
      <td>11</td>
      <td>-0.351</td>
      <td>343.06</td>
      <td>1677.10</td>
      <td>305.263</td>
      <td>205.893</td>
      <td>371.770</td>
      <td>1365.6802</td>
      <td>0.0202</td>
      <td>-0.123</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1984</td>
      <td>12</td>
      <td>-0.611</td>
      <td>344.54</td>
      <td>1672.15</td>
      <td>305.313</td>
      <td>207.308</td>
      <td>372.701</td>
      <td>1365.7617</td>
      <td>0.0188</td>
      <td>-0.282</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1985</td>
      <td>1</td>
      <td>-0.561</td>
      <td>345.25</td>
      <td>1663.42</td>
      <td>305.301</td>
      <td>208.537</td>
      <td>373.623</td>
      <td>1365.6082</td>
      <td>0.0164</td>
      <td>-0.001</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1985</td>
      <td>2</td>
      <td>-0.602</td>
      <td>346.06</td>
      <td>1666.21</td>
      <td>305.243</td>
      <td>209.543</td>
      <td>374.681</td>
      <td>1365.7085</td>
      <td>0.0160</td>
      <td>-0.155</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1985</td>
      <td>3</td>
      <td>-0.737</td>
      <td>347.66</td>
      <td>1678.34</td>
      <td>305.165</td>
      <td>210.368</td>
      <td>376.004</td>
      <td>1365.6570</td>
      <td>0.0141</td>
      <td>-0.032</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1985</td>
      <td>4</td>
      <td>-0.484</td>
      <td>348.20</td>
      <td>1675.24</td>
      <td>305.093</td>
      <td>211.111</td>
      <td>377.635</td>
      <td>1365.5120</td>
      <td>0.0138</td>
      <td>-0.042</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1985</td>
      <td>5</td>
      <td>-0.731</td>
      <td>348.92</td>
      <td>1666.83</td>
      <td>305.045</td>
      <td>211.823</td>
      <td>379.539</td>
      <td>1365.6366</td>
      <td>0.0128</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1985</td>
      <td>6</td>
      <td>-0.086</td>
      <td>348.40</td>
      <td>1659.40</td>
      <td>305.027</td>
      <td>212.512</td>
      <td>381.642</td>
      <td>1365.6964</td>
      <td>0.0126</td>
      <td>-0.049</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1985</td>
      <td>7</td>
      <td>-0.156</td>
      <td>346.66</td>
      <td>1654.25</td>
      <td>305.049</td>
      <td>213.165</td>
      <td>383.905</td>
      <td>1365.6509</td>
      <td>0.0121</td>
      <td>-0.042</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1985</td>
      <td>8</td>
      <td>-0.392</td>
      <td>344.85</td>
      <td>1654.41</td>
      <td>305.126</td>
      <td>213.803</td>
      <td>386.223</td>
      <td>1365.7499</td>
      <td>0.0116</td>
      <td>0.013</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1985</td>
      <td>9</td>
      <td>-0.541</td>
      <td>343.20</td>
      <td>1668.31</td>
      <td>305.250</td>
      <td>214.501</td>
      <td>388.500</td>
      <td>1365.6653</td>
      <td>0.0102</td>
      <td>-0.035</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1985</td>
      <td>10</td>
      <td>-0.140</td>
      <td>343.08</td>
      <td>1681.56</td>
      <td>305.395</td>
      <td>215.327</td>
      <td>390.676</td>
      <td>1365.5269</td>
      <td>0.0101</td>
      <td>-0.008</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>278</th>
      <td>2006</td>
      <td>7</td>
      <td>0.628</td>
      <td>382.38</td>
      <td>1765.95</td>
      <td>319.872</td>
      <td>249.247</td>
      <td>539.725</td>
      <td>1365.8212</td>
      <td>0.0038</td>
      <td>0.456</td>
    </tr>
    <tr>
      <th>279</th>
      <td>2006</td>
      <td>8</td>
      <td>0.759</td>
      <td>380.45</td>
      <td>1762.66</td>
      <td>319.930</td>
      <td>248.981</td>
      <td>539.682</td>
      <td>1365.7067</td>
      <td>0.0041</td>
      <td>0.482</td>
    </tr>
    <tr>
      <th>280</th>
      <td>2006</td>
      <td>9</td>
      <td>0.793</td>
      <td>378.92</td>
      <td>1776.04</td>
      <td>320.010</td>
      <td>248.775</td>
      <td>539.566</td>
      <td>1365.8419</td>
      <td>0.0043</td>
      <td>0.425</td>
    </tr>
    <tr>
      <th>281</th>
      <td>2006</td>
      <td>10</td>
      <td>0.892</td>
      <td>379.16</td>
      <td>1789.02</td>
      <td>320.125</td>
      <td>248.666</td>
      <td>539.488</td>
      <td>1365.8270</td>
      <td>0.0044</td>
      <td>0.472</td>
    </tr>
    <tr>
      <th>282</th>
      <td>2006</td>
      <td>11</td>
      <td>1.292</td>
      <td>380.18</td>
      <td>1791.91</td>
      <td>320.321</td>
      <td>248.605</td>
      <td>539.500</td>
      <td>1365.7039</td>
      <td>0.0049</td>
      <td>0.440</td>
    </tr>
    <tr>
      <th>283</th>
      <td>2006</td>
      <td>12</td>
      <td>0.951</td>
      <td>381.79</td>
      <td>1795.04</td>
      <td>320.451</td>
      <td>248.480</td>
      <td>539.377</td>
      <td>1365.7087</td>
      <td>0.0054</td>
      <td>0.518</td>
    </tr>
    <tr>
      <th>284</th>
      <td>2007</td>
      <td>1</td>
      <td>0.974</td>
      <td>382.93</td>
      <td>1799.66</td>
      <td>320.561</td>
      <td>248.372</td>
      <td>539.206</td>
      <td>1365.7173</td>
      <td>0.0054</td>
      <td>0.601</td>
    </tr>
    <tr>
      <th>285</th>
      <td>2007</td>
      <td>2</td>
      <td>0.510</td>
      <td>383.81</td>
      <td>1803.08</td>
      <td>320.571</td>
      <td>248.264</td>
      <td>538.973</td>
      <td>1365.7145</td>
      <td>0.0051</td>
      <td>0.498</td>
    </tr>
    <tr>
      <th>286</th>
      <td>2007</td>
      <td>3</td>
      <td>0.074</td>
      <td>384.56</td>
      <td>1803.10</td>
      <td>320.548</td>
      <td>247.997</td>
      <td>538.811</td>
      <td>1365.7544</td>
      <td>0.0045</td>
      <td>0.435</td>
    </tr>
    <tr>
      <th>287</th>
      <td>2007</td>
      <td>4</td>
      <td>-0.049</td>
      <td>386.40</td>
      <td>1802.11</td>
      <td>320.518</td>
      <td>247.574</td>
      <td>538.586</td>
      <td>1365.7228</td>
      <td>0.0045</td>
      <td>0.466</td>
    </tr>
    <tr>
      <th>288</th>
      <td>2007</td>
      <td>5</td>
      <td>0.183</td>
      <td>386.58</td>
      <td>1795.65</td>
      <td>320.445</td>
      <td>247.224</td>
      <td>538.130</td>
      <td>1365.6932</td>
      <td>0.0041</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>289</th>
      <td>2007</td>
      <td>6</td>
      <td>-0.358</td>
      <td>386.05</td>
      <td>1781.81</td>
      <td>320.332</td>
      <td>246.881</td>
      <td>537.376</td>
      <td>1365.7616</td>
      <td>0.0040</td>
      <td>0.382</td>
    </tr>
    <tr>
      <th>290</th>
      <td>2007</td>
      <td>7</td>
      <td>-0.290</td>
      <td>384.49</td>
      <td>1771.89</td>
      <td>320.349</td>
      <td>246.497</td>
      <td>537.113</td>
      <td>1365.7506</td>
      <td>0.0040</td>
      <td>0.394</td>
    </tr>
    <tr>
      <th>291</th>
      <td>2007</td>
      <td>8</td>
      <td>-0.440</td>
      <td>382.00</td>
      <td>1779.38</td>
      <td>320.471</td>
      <td>246.307</td>
      <td>537.125</td>
      <td>1365.7566</td>
      <td>0.0041</td>
      <td>0.358</td>
    </tr>
    <tr>
      <th>292</th>
      <td>2007</td>
      <td>9</td>
      <td>-1.162</td>
      <td>380.90</td>
      <td>1794.21</td>
      <td>320.618</td>
      <td>246.214</td>
      <td>537.281</td>
      <td>1365.7159</td>
      <td>0.0042</td>
      <td>0.402</td>
    </tr>
    <tr>
      <th>293</th>
      <td>2007</td>
      <td>10</td>
      <td>-1.142</td>
      <td>381.14</td>
      <td>1802.38</td>
      <td>320.855</td>
      <td>246.189</td>
      <td>537.380</td>
      <td>1365.7388</td>
      <td>0.0041</td>
      <td>0.362</td>
    </tr>
    <tr>
      <th>294</th>
      <td>2007</td>
      <td>11</td>
      <td>-1.177</td>
      <td>382.42</td>
      <td>1803.79</td>
      <td>321.062</td>
      <td>246.178</td>
      <td>537.319</td>
      <td>1365.6680</td>
      <td>0.0042</td>
      <td>0.266</td>
    </tr>
    <tr>
      <th>295</th>
      <td>2007</td>
      <td>12</td>
      <td>-1.168</td>
      <td>383.89</td>
      <td>1805.58</td>
      <td>321.217</td>
      <td>246.261</td>
      <td>537.052</td>
      <td>1365.6927</td>
      <td>0.0040</td>
      <td>0.226</td>
    </tr>
    <tr>
      <th>296</th>
      <td>2008</td>
      <td>1</td>
      <td>-1.011</td>
      <td>385.44</td>
      <td>1809.92</td>
      <td>321.328</td>
      <td>246.183</td>
      <td>536.876</td>
      <td>1365.7163</td>
      <td>0.0038</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>297</th>
      <td>2008</td>
      <td>2</td>
      <td>-1.402</td>
      <td>385.73</td>
      <td>1803.45</td>
      <td>321.345</td>
      <td>245.898</td>
      <td>536.484</td>
      <td>1365.7366</td>
      <td>0.0036</td>
      <td>0.198</td>
    </tr>
    <tr>
      <th>298</th>
      <td>2008</td>
      <td>3</td>
      <td>-1.635</td>
      <td>385.97</td>
      <td>1792.84</td>
      <td>321.295</td>
      <td>245.430</td>
      <td>535.979</td>
      <td>1365.6726</td>
      <td>0.0034</td>
      <td>0.447</td>
    </tr>
    <tr>
      <th>299</th>
      <td>2008</td>
      <td>4</td>
      <td>-0.942</td>
      <td>387.16</td>
      <td>1792.57</td>
      <td>321.354</td>
      <td>245.086</td>
      <td>535.648</td>
      <td>1365.7146</td>
      <td>0.0033</td>
      <td>0.278</td>
    </tr>
    <tr>
      <th>300</th>
      <td>2008</td>
      <td>5</td>
      <td>-0.355</td>
      <td>388.50</td>
      <td>1796.43</td>
      <td>321.420</td>
      <td>244.914</td>
      <td>535.399</td>
      <td>1365.7175</td>
      <td>0.0031</td>
      <td>0.283</td>
    </tr>
    <tr>
      <th>301</th>
      <td>2008</td>
      <td>6</td>
      <td>0.128</td>
      <td>387.88</td>
      <td>1791.80</td>
      <td>321.447</td>
      <td>244.676</td>
      <td>535.128</td>
      <td>1365.6730</td>
      <td>0.0031</td>
      <td>0.315</td>
    </tr>
    <tr>
      <th>302</th>
      <td>2008</td>
      <td>7</td>
      <td>0.003</td>
      <td>386.42</td>
      <td>1782.93</td>
      <td>321.372</td>
      <td>244.434</td>
      <td>535.026</td>
      <td>1365.6720</td>
      <td>0.0033</td>
      <td>0.406</td>
    </tr>
    <tr>
      <th>303</th>
      <td>2008</td>
      <td>8</td>
      <td>-0.266</td>
      <td>384.15</td>
      <td>1779.88</td>
      <td>321.405</td>
      <td>244.200</td>
      <td>535.072</td>
      <td>1365.6570</td>
      <td>0.0036</td>
      <td>0.407</td>
    </tr>
    <tr>
      <th>304</th>
      <td>2008</td>
      <td>9</td>
      <td>-0.643</td>
      <td>383.09</td>
      <td>1795.08</td>
      <td>321.529</td>
      <td>244.083</td>
      <td>535.048</td>
      <td>1365.6647</td>
      <td>0.0043</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>305</th>
      <td>2008</td>
      <td>10</td>
      <td>-0.780</td>
      <td>382.99</td>
      <td>1814.18</td>
      <td>321.796</td>
      <td>244.080</td>
      <td>534.927</td>
      <td>1365.6759</td>
      <td>0.0046</td>
      <td>0.440</td>
    </tr>
    <tr>
      <th>306</th>
      <td>2008</td>
      <td>11</td>
      <td>-0.621</td>
      <td>384.13</td>
      <td>1812.37</td>
      <td>322.013</td>
      <td>244.225</td>
      <td>534.906</td>
      <td>1365.7065</td>
      <td>0.0048</td>
      <td>0.394</td>
    </tr>
    <tr>
      <th>307</th>
      <td>2008</td>
      <td>12</td>
      <td>-0.666</td>
      <td>385.56</td>
      <td>1812.88</td>
      <td>322.182</td>
      <td>244.204</td>
      <td>535.005</td>
      <td>1365.6926</td>
      <td>0.0046</td>
      <td>0.330</td>
    </tr>
  </tbody>
</table>
<p>308 rows × 11 columns</p>
</div>




```python
climate_change_1_train=climate_change_1.iloc[0:284]
#climate_change_1_train
climate_change_1_test=climate_change_1.iloc[284:308]
#climate_change_1_test
```


```python
def closed_form_1(df: pd.core.frame.DataFrame, column: int = 10)-> np.ndarray:
    X = df.drop(df.columns[column],axis=1).to_numpy()
    X = np.concatenate([np.ones((len(X),1)),X],axis = 1)
    # X: the features
    Y = df.iloc[:,[column]].to_numpy()
    Y = Y.reshape((len(Y)))
    # Y: the results
    theta = inv(X.T @ X)@ X.T @ Y
    return theta

def closed_form_2(X:np.ndarray, Y:np.ndarray)-> np.ndarray:
    theta = inv(X.T @ X)@ X.T @ Y
    return theta

closed_form_1(climate_change_1_train)[1:]
```




    array([ 8.23965077e-03, -3.61230802e-03,  6.44651665e-02,  2.51139083e-03,
            1.87173456e-04, -1.63078511e-02, -6.27428702e-03,  3.42652458e-03,
            9.51816184e-02, -1.54295992e+00])




```python
# Using scipy to check:
from sklearn.linear_model import LinearRegression as lm
X=climate_change_1_train.drop(climate_change_1_train.columns[10],axis=1).to_numpy()
Y=climate_change_1_train.iloc[:,[10]].to_numpy()
l=lm().fit(X,Y)
l.coef_
```




    array([[ 8.23964892e-03, -3.61230811e-03,  6.44651665e-02,
             2.51139097e-03,  1.87173443e-04, -1.63078496e-02,
            -6.27428715e-03,  3.42652470e-03,  9.51816165e-02,
            -1.54295993e+00]])



# 2.R2

Q: Write down the mathematical formula for the linear model and evaluate the model R2 on the training set and the testing set.


```python
r_sq = l.score(X, Y)
r_sq
```




    0.7549422940386257



# 3.Significant Variables

Q: Which variables are significant in the model?


```python
import statsmodels.api as sm
mod = sm.OLS(Y,X)
fit = mod.fit()
p_values = fit.summary2().tables[1]['P>|t|']
p_values
#MEI,CO2,N2O,CFC-11,CFC-12,TSI,Aerosols are significant in the model(0.05 significant level).
```




    x1     1.594419e-117
    x2      5.053259e-14
    x3      3.179871e-01
    x4      6.826122e-16
    x5      0.000000e+00
    x6      2.880761e-69
    x7      6.325787e-28
    x8      4.579785e-45
    x9      3.009168e-76
    x10     7.795461e-02
    x11     1.066327e-02
    Name: P>|t|, dtype: float64



# 4. For climate_change_2.csv

Q: Write down the necessary conditions for using the closed form solution. And you can apply it to the dataset climate_change_2.csv, explain the solution is unreasonable.


```python
climate_change_2 = pd.read_csv('climate_change_2.csv')
#climate_change_2
```


```python
climate_change_2_train=climate_change_2.iloc[0:284]
climate_change_2_test=climate_change_2.iloc[284:308]
closed_form_1(climate_change_2_train)[1:]
```




    array([ 2.01083594e-12, -5.36251599e-12,  5.10582687e-11, -1.28112521e-11,
            1.00000000e-03,  5.87157545e-11,  4.17712392e-12, -3.05891423e-12,
            2.25990338e-11,  1.08553680e-10, -1.01428310e-11])




```python
from sklearn.linear_model import LinearRegression as lm
X=climate_change_2_train.drop(climate_change_2_train.columns[10],axis=1).to_numpy()
Y=climate_change_2_train.iloc[:,[10]].to_numpy()
l=lm().fit(X,Y)
l.coef_
```




    array([[ 1.12051329e-17,  6.66402228e-18,  1.04772243e-17,
             7.73061207e-18,  1.00000000e-03, -1.08634025e-17,
             7.13888333e-18, -6.95792118e-18, -1.76950359e-17,
             1.12119206e-16,  1.91583910e-16]])




```python
import pandas as pd
climate_change_2_corr = climate_change_2.corr()
# Visualization
import matplotlib.pyplot as mp, seaborn
seaborn.heatmap(climate_change_2_corr, center=0, annot=True)
mp.show()
```


![png](output_19_0.png)


It can be concluded from the correlation matrix that NO and CH4 are completely linearly correlated, so there is no inverse matrix, and the formula is invalid. So the solution is unreasonable.

# Problem 2----Regularization

# 1.Loss Function

Q: Please write down the loss function for linear model with L1 regularization, L2 regularization, respectively.


```python
#L1 regularization
def L1Norm(l, theta):
    return  np.dot(np.abs(theta), np.ones(theta.size)) * l
 
def L1NormPartial(l, theta):
    return np.sign(theta) * l

# For linear regression, the derivative of J function is:
def __Jfunction(self):        
    sum = 0
    for i in range(0, self.m):
        err = self.__error_dist(self.x[i], self.y[i])
        sum += np.dot(err, err)
        sum += Regularization.L2Norm(0.8, self.theta)
        return 1/(2 * self.m) * sum
```


```python
#L2 regularization
def L2Norm(l, theta):
    return  np.dot(theta, theta) * l 
 
def L2NormPartial(l, theta):
    return theta * l

# For linear regression, the derivative of J function is:
def __partialderiv_J_func(self):
        sum = 0
        for i in range(0, self.m):
            err = self.__error_dist(self.x[i], self.y[i])
            sum += np.dot(self.x[i], err)
            sum += Regularization.L2NormPartial(0.8, self.theta)
            return 1/self.m * sum
```

# 2.Closed Form Solution

Q: The closed form solution for linear model with L2 regularization:
𝛉 = (𝐗𝐓𝐗 + 𝛌𝐈)−𝟏𝐗𝐓𝐘
where I is the identity matrix. Write a function closed_form_2 that computes this closed form solution given the features X, labels Y and the regularization parameter λ.

We can answer questions 2 and 4 together.


```python
def closed_form_2():

    dataset = pd.read_csv("climate_change_1.csv")
    X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

    y = dataset.get("Temp")
  
    X = np.column_stack((X,np.ones(len(X))))

    for lambda1 in [10,1,0.1,0.01,0.001]:
        X_train = X[:284]
        X_test = X[284:]
        y_train = y[:284]
        y_test = y[284:]
    
        X_train=np.mat(X_train)
        y_train = np.mat(y_train).T
        xTx = X_train.T*X_train
        w = 0
        print("="*25+"L2 Regularization (lambda is "+str(lambda1)+")"+"="*25)
        I_m= np.eye(X_train.shape[1])
        if np.linalg.det(xTx+lambda1*I_m)==0.0:
            print("xTx is invertible")
        else:
            print(np.linalg.det(xTx+lambda1*I_m))
            w= (xTx+lambda1*I_m).I*(X_train.T*y_train)
        wights = np.ravel(w)    
        y_train_pred = np.ravel(np.mat(X_train)*np.mat(w))
        y_test_pred = np.ravel(np.mat(X_test)*np.mat(w))
        coef_=wights[:-1]
        intercept_=wights[-1]

        X_train = np.ravel(X_train).reshape(-1,9)
        y_train = np.ravel(y_train)
        
        print("Coefficient: ",coef_)
        print("Intercept: ",intercept_)
        print("the model is： y = ",coef_,"* X +(",intercept_,")")
        y_train_avg = np.average(y_train)
    
        R2_train = 1-(np.average((y_train-y_train_pred)**2))/(np.average((y_train-y_train_avg)**2))
        print("R2 in Train ： ",R2_train)
     
        y_test_avg = np.average(y_test)
        R2_test = 1-(np.average((y_test-y_test_pred)**2))/(np.average((y_test-y_test_avg)**2))
        print("R2 in Test ： ",R2_test)

closed_form_2()
```

    =========================L2 Regularization (lambda is 10)=========================
    4.052005289688253e+33
    Coefficient:  [ 0.04054315  0.00814554  0.00020508 -0.01608137 -0.00636145  0.003689
      0.00126458 -0.02443305]
    Intercept:  -0.00022022058288633274
    the model is： y =  [ 0.04054315  0.00814554  0.00020508 -0.01608137 -0.00636145  0.003689
      0.00126458 -0.02443305] * X +( -0.00022022058288633274 )
    R2 in Train ：  0.6803719394071281
    R2 in Test ：  -0.7061640575416965
    =========================L2 Regularization (lambda is 1)=========================
    4.182558175861993e+31
    Coefficient:  [ 0.04395558  0.00804313  0.00021395 -0.01693027 -0.00646627  0.00376881
      0.00146759 -0.21177258]
    Intercept:  -0.0022945422838525635
    the model is： y =  [ 0.04395558  0.00804313  0.00021395 -0.01693027 -0.00646627  0.00376881
      0.00146759 -0.21177258] * X +( -0.0022945422838525635 )
    R2 in Train ：  0.6897571586198687
    R2 in Test ：  -0.5861726468586046
    =========================L2 Regularization (lambda is 0.1)=========================
    1.0051083854786037e+30
    Coefficient:  [ 5.06851277e-02  6.98925378e-03  1.30761990e-04 -1.48156599e-02
     -6.07864608e-03  3.66100278e-03  1.36118274e-03 -8.71332452e-01]
    Intercept:  -0.025045661913281534
    the model is： y =  [ 5.06851277e-02  6.98925378e-03  1.30761990e-04 -1.48156599e-02
     -6.07864608e-03  3.66100278e-03  1.36118274e-03 -8.71332452e-01] * X +( -0.025045661913281534 )
    R2 in Train ：  0.7110310866063567
    R2 in Test ：  -0.36213522139292387
    =========================L2 Regularization (lambda is 0.01)=========================
    6.930175866500259e+28
    Coefficient:  [ 5.46344723e-02  6.35012916e-03  7.94610956e-05 -1.34794077e-02
     -5.83699154e-03  3.59093203e-03  1.44947810e-03 -1.26505174e+00]
    Intercept:  -0.26232414556713424
    the model is： y =  [ 5.46344723e-02  6.35012916e-03  7.94610956e-05 -1.34794077e-02
     -5.83699154e-03  3.59093203e-03  1.44947810e-03 -1.26505174e+00] * X +( -0.26232414556713424 )
    R2 in Train ：  0.7153953027375966
    R2 in Test ：  -0.24446585990645597
    =========================L2 Regularization (lambda is 0.001)=========================
    6.742979655964518e+27
    Coefficient:  [ 5.53981612e-02  6.25686043e-03  7.26293229e-05 -1.33359358e-02
     -5.81554289e-03  3.58444627e-03  3.15485922e-03 -1.32868779e+00]
    Intercept:  -2.5924696366355677
    the model is： y =  [ 5.53981612e-02  6.25686043e-03  7.26293229e-05 -1.33359358e-02
     -5.81554289e-03  3.58444627e-03  3.15485922e-03 -1.32868779e+00] * X +( -2.5924696366355677 )
    R2 in Train ：  0.7168000467674187
    R2 in Test ：  -0.21604536980238898


# 3.Comparasion

Q: Compare the two solutions in problem 1 and problem 2 and explain the reason why linear model with L2 regularization is robust. (using climate_change_1.csv)

It will reduce the coefficient of unimportant prediction factors close to 0 and avoid overfitting. In L2 model, it is less sensitive to single variable, so it is more robust.

# 4.Change the regularization parameter λ

Q: You can change the regularization parameter λ to get different solutions for this problem. Suppose we set λ = 10, 1, 0.1, 0.01, 0.001, and please evaluate the model R2 on the training set and the testing set. Finally, please decide the best regularization parameter λ. (Note that: As a qualified data analyst, you must know how to choose model parameters, please learn about cross validation methods.)

The anwser can see the above(in Q2).

# Problem 3 — Feature Selection

# 1.Workflow

Q: From Problem 1, you can know which variables are significant, therefore you can use less variables to train model. For example, remove highly correlated and redundant features. You can propose a workflow to select feature.

Solution:
For m features, from k=1 to k = m:
We can choose k features from m features, and establish C (m, K) models, then choose the best one (MSE minimum or R2 maximum);
Then select an optimal model from the m optimal models.

# 2.Better Model

Train a better model than the model in Problem 2.


```python
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model

#Variance Inflation Factor
def vif(X, thres=10.0):
    col = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:,col].values, ix) for ix in range(X.iloc[:,col].shape[1])]
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            del col[maxix]
            print('delete=',X.columns[col[maxix]],'  ', 'vif=',maxvif )
            dropped = True
    print('Remain Variables:', list(X.columns[col]))
    print('VIF:', vif)
    return list(X.columns[col]) 

dataset = pd.read_csv("climate_change_1.csv")
X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

y = dataset.get("Temp")

X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]
d = vif(X_train)
print(d)

X = dataset.get( ['MEI', 'CFC-12', 'Aerosols'])
y = dataset.get("Temp")
X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
print('coefficients(b1,b2...):',regr.coef_)
print('intercept(b0):',regr.intercept_)
y_train_pred = regr.predict(X_train)
       
R2_1 = regr.score(X_train, y_train)
print(R2_1)
R2_2 = regr.score(X_test, y_test)
print(R2_2)
```

    delete= CFC-11    vif= 239743.2424704495
    delete= Aerosols    vif= 29867.18540477364
    delete= CFC-11    vif= 11884.79599294173
    delete= CFC-12    vif= 502.06957361985695
    delete= CFC-12    vif= 122.31236225671839
    Remain Variables: ['MEI', 'CFC-12', 'Aerosols']
    VIF: [1.2888871669460935, 1.33868239281389, 1.48103752609454]
    ['MEI', 'CFC-12', 'Aerosols']
    coefficients(b1,b2...): [ 5.54993375e-02  1.86365387e-03 -2.08242114e+00]
    intercept(b0): -0.6553255026654846
    0.5996443150479794
    0.004717686204946725


# Problem 4 — Gradient Descent

Gradient descent algorithm is an iterative process that takes us to the minimum of a function. Please write down the iterative expression for updating the solution of linear model and implement it using Python or Matlab in gradientDescent function.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def costFunc(X,Y,theta):
    #cost func
    inner=np.power((X*theta.T)-Y,2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X,Y,theta,alpha,iters):
    temp = np.mat(np.zeros(theta.shape))
    cost = np.zeros(iters)
    thetaNums = int(theta.shape[1])
    
    for i in range(iters):
        error = (X*theta.T-Y)
        for j in range(thetaNums):
            derivativeInner = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j]-(alpha*np.sum(derivativeInner)/len(X))
        theta = temp
        cost[i]=costFunc(X,Y,theta)
    return theta,cost


dataset = pd.read_csv("climate_change_1.csv")
X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

y = dataset.get("Temp")
X = np.column_stack((np.ones(len(X)),X))
X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]

X_train = np.mat(X_train)  
Y_train = np.mat(y_train).T

for i in range(1,9):
    X_train[:,i] = (X_train[:,i] - min(X_train[:,i])) / (max(X_train[:,i]) - min(X_train[:,i]))

theta_n = (X_train.T*X_train).I*X_train.T*Y_train
print("theta =",theta_n)
theta = np.mat([0,0,0,0,0,0,0,0,0])
iters = 100000
alpha = 0.001

finalTheta,cost = gradientDescent(X_train,Y_train,theta,alpha,iters)
print("final theta ",finalTheta)
print("cost ",cost)

fig, bx = plt.subplots(figsize=(8,6))
bx.plot(np.arange(iters), cost, 'r') 
bx.set_xlabel('Iterations') 
bx.set_ylabel('Cost') 
bx.set_title('Error vs. Training Epoch') 
plt.show()
```

    theta = [[-0.07698894]
     [ 0.29450977]
     [ 0.28935427]
     [ 0.02211171]
     [-0.27724073]
     [-0.53156629]
     [ 0.7376296 ]
     [ 0.17604596]
     [-0.22725924]]
    final theta  [[-0.09315388  0.26327692  0.20584575  0.05590722  0.1773908  -0.10907193
       0.09177624  0.13999486 -0.2096555 ]]
    cost  [0.04678781 0.04652792 0.04626986 ... 0.00428416 0.00428416 0.00428416]



![png](output_45_1.png)

