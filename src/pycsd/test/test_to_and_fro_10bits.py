#! /usr/bin/env python3
"""
 Unittests for the CSD module
"""

import unittest

import pycsd.csd as csd

# generated by recursive alg in ../dev/csd_numbers.py
good_values_dict = {
    0.0: "0000000000",
    1.0: "000000000+",
    2.0: "00000000+0",
    3.0: "0000000+0-",
    4.0: "0000000+00",
    5.0: "0000000+0+",
    6.0: "000000+0-0",
    7.0: "000000+00-",
    8.0: "000000+000",
    9.0: "000000+00+",
    10.0: "000000+0+0",
    11.0: "00000+0-0-",
    12.0: "00000+0-00",
    13.0: "00000+0-0+",
    14.0: "00000+00-0",
    15.0: "00000+000-",
    16.0: "00000+0000",
    17.0: "00000+000+",
    18.0: "00000+00+0",
    19.0: "00000+0+0-",
    20.0: "00000+0+00",
    21.0: "00000+0+0+",
    22.0: "0000+0-0-0",
    23.0: "0000+0-00-",
    24.0: "0000+0-000",
    25.0: "0000+0-00+",
    26.0: "0000+0-0+0",
    27.0: "0000+00-0-",
    28.0: "0000+00-00",
    29.0: "0000+00-0+",
    30.0: "0000+000-0",
    31.0: "0000+0000-",
    32.0: "0000+00000",
    33.0: "0000+0000+",
    34.0: "0000+000+0",
    35.0: "0000+00+0-",
    36.0: "0000+00+00",
    37.0: "0000+00+0+",
    38.0: "0000+0+0-0",
    39.0: "0000+0+00-",
    40.0: "0000+0+000",
    41.0: "0000+0+00+",
    42.0: "0000+0+0+0",
    43.0: "000+0-0-0-",
    44.0: "000+0-0-00",
    45.0: "000+0-0-0+",
    46.0: "000+0-00-0",
    47.0: "000+0-000-",
    48.0: "000+0-0000",
    49.0: "000+0-000+",
    50.0: "000+0-00+0",
    51.0: "000+0-0+0-",
    52.0: "000+0-0+00",
    53.0: "000+0-0+0+",
    54.0: "000+00-0-0",
    55.0: "000+00-00-",
    56.0: "000+00-000",
    57.0: "000+00-00+",
    58.0: "000+00-0+0",
    59.0: "000+000-0-",
    60.0: "000+000-00",
    61.0: "000+000-0+",
    62.0: "000+0000-0",
    63.0: "000+00000-",
    64.0: "000+000000",
    65.0: "000+00000+",
    66.0: "000+0000+0",
    67.0: "000+000+0-",
    68.0: "000+000+00",
    69.0: "000+000+0+",
    70.0: "000+00+0-0",
    71.0: "000+00+00-",
    72.0: "000+00+000",
    73.0: "000+00+00+",
    74.0: "000+00+0+0",
    75.0: "000+0+0-0-",
    76.0: "000+0+0-00",
    77.0: "000+0+0-0+",
    78.0: "000+0+00-0",
    79.0: "000+0+000-",
    80.0: "000+0+0000",
    81.0: "000+0+000+",
    82.0: "000+0+00+0",
    83.0: "000+0+0+0-",
    84.0: "000+0+0+00",
    85.0: "000+0+0+0+",
    86.0: "00+0-0-0-0",
    87.0: "00+0-0-00-",
    88.0: "00+0-0-000",
    89.0: "00+0-0-00+",
    90.0: "00+0-0-0+0",
    91.0: "00+0-00-0-",
    92.0: "00+0-00-00",
    93.0: "00+0-00-0+",
    94.0: "00+0-000-0",
    95.0: "00+0-0000-",
    96.0: "00+0-00000",
    97.0: "00+0-0000+",
    98.0: "00+0-000+0",
    99.0: "00+0-00+0-",
    100.0: "00+0-00+00",
    101.0: "00+0-00+0+",
    102.0: "00+0-0+0-0",
    103.0: "00+0-0+00-",
    104.0: "00+0-0+000",
    105.0: "00+0-0+00+",
    106.0: "00+0-0+0+0",
    107.0: "00+00-0-0-",
    108.0: "00+00-0-00",
    109.0: "00+00-0-0+",
    110.0: "00+00-00-0",
    111.0: "00+00-000-",
    112.0: "00+00-0000",
    113.0: "00+00-000+",
    114.0: "00+00-00+0",
    115.0: "00+00-0+0-",
    116.0: "00+00-0+00",
    117.0: "00+00-0+0+",
    118.0: "00+000-0-0",
    119.0: "00+000-00-",
    120.0: "00+000-000",
    121.0: "00+000-00+",
    122.0: "00+000-0+0",
    123.0: "00+0000-0-",
    124.0: "00+0000-00",
    125.0: "00+0000-0+",
    126.0: "00+00000-0",
    127.0: "00+000000-",
    128.0: "00+0000000",
    129.0: "00+000000+",
    130.0: "00+00000+0",
    131.0: "00+0000+0-",
    132.0: "00+0000+00",
    133.0: "00+0000+0+",
    134.0: "00+000+0-0",
    135.0: "00+000+00-",
    136.0: "00+000+000",
    137.0: "00+000+00+",
    138.0: "00+000+0+0",
    139.0: "00+00+0-0-",
    140.0: "00+00+0-00",
    141.0: "00+00+0-0+",
    142.0: "00+00+00-0",
    143.0: "00+00+000-",
    144.0: "00+00+0000",
    145.0: "00+00+000+",
    146.0: "00+00+00+0",
    147.0: "00+00+0+0-",
    148.0: "00+00+0+00",
    149.0: "00+00+0+0+",
    150.0: "00+0+0-0-0",
    151.0: "00+0+0-00-",
    152.0: "00+0+0-000",
    153.0: "00+0+0-00+",
    154.0: "00+0+0-0+0",
    155.0: "00+0+00-0-",
    156.0: "00+0+00-00",
    157.0: "00+0+00-0+",
    158.0: "00+0+000-0",
    159.0: "00+0+0000-",
    160.0: "00+0+00000",
    161.0: "00+0+0000+",
    162.0: "00+0+000+0",
    163.0: "00+0+00+0-",
    164.0: "00+0+00+00",
    165.0: "00+0+00+0+",
    166.0: "00+0+0+0-0",
    167.0: "00+0+0+00-",
    168.0: "00+0+0+000",
    169.0: "00+0+0+00+",
    170.0: "00+0+0+0+0",
    171.0: "0+0-0-0-0-",
    172.0: "0+0-0-0-00",
    173.0: "0+0-0-0-0+",
    174.0: "0+0-0-00-0",
    175.0: "0+0-0-000-",
    176.0: "0+0-0-0000",
    177.0: "0+0-0-000+",
    178.0: "0+0-0-00+0",
    179.0: "0+0-0-0+0-",
    180.0: "0+0-0-0+00",
    181.0: "0+0-0-0+0+",
    182.0: "0+0-00-0-0",
    183.0: "0+0-00-00-",
    184.0: "0+0-00-000",
    185.0: "0+0-00-00+",
    186.0: "0+0-00-0+0",
    187.0: "0+0-000-0-",
    188.0: "0+0-000-00",
    189.0: "0+0-000-0+",
    190.0: "0+0-0000-0",
    191.0: "0+0-00000-",
    192.0: "0+0-000000",
    193.0: "0+0-00000+",
    194.0: "0+0-0000+0",
    195.0: "0+0-000+0-",
    196.0: "0+0-000+00",
    197.0: "0+0-000+0+",
    198.0: "0+0-00+0-0",
    199.0: "0+0-00+00-",
    200.0: "0+0-00+000",
    201.0: "0+0-00+00+",
    202.0: "0+0-00+0+0",
    203.0: "0+0-0+0-0-",
    204.0: "0+0-0+0-00",
    205.0: "0+0-0+0-0+",
    206.0: "0+0-0+00-0",
    207.0: "0+0-0+000-",
    208.0: "0+0-0+0000",
    209.0: "0+0-0+000+",
    210.0: "0+0-0+00+0",
    211.0: "0+0-0+0+0-",
    212.0: "0+0-0+0+00",
    213.0: "0+0-0+0+0+",
    214.0: "0+00-0-0-0",
    215.0: "0+00-0-00-",
    216.0: "0+00-0-000",
    217.0: "0+00-0-00+",
    218.0: "0+00-0-0+0",
    219.0: "0+00-00-0-",
    220.0: "0+00-00-00",
    221.0: "0+00-00-0+",
    222.0: "0+00-000-0",
    223.0: "0+00-0000-",
    224.0: "0+00-00000",
    225.0: "0+00-0000+",
    226.0: "0+00-000+0",
    227.0: "0+00-00+0-",
    228.0: "0+00-00+00",
    229.0: "0+00-00+0+",
    230.0: "0+00-0+0-0",
    231.0: "0+00-0+00-",
    232.0: "0+00-0+000",
    233.0: "0+00-0+00+",
    234.0: "0+00-0+0+0",
    235.0: "0+000-0-0-",
    236.0: "0+000-0-00",
    237.0: "0+000-0-0+",
    238.0: "0+000-00-0",
    239.0: "0+000-000-",
    240.0: "0+000-0000",
    241.0: "0+000-000+",
    242.0: "0+000-00+0",
    243.0: "0+000-0+0-",
    244.0: "0+000-0+00",
    245.0: "0+000-0+0+",
    246.0: "0+0000-0-0",
    247.0: "0+0000-00-",
    248.0: "0+0000-000",
    249.0: "0+0000-00+",
    250.0: "0+0000-0+0",
    251.0: "0+00000-0-",
    252.0: "0+00000-00",
    253.0: "0+00000-0+",
    254.0: "0+000000-0",
    255.0: "0+0000000-",
    256.0: "0+00000000",
    257.0: "0+0000000+",
    258.0: "0+000000+0",
    259.0: "0+00000+0-",
    260.0: "0+00000+00",
    261.0: "0+00000+0+",
    262.0: "0+0000+0-0",
    263.0: "0+0000+00-",
    264.0: "0+0000+000",
    265.0: "0+0000+00+",
    266.0: "0+0000+0+0",
    267.0: "0+000+0-0-",
    268.0: "0+000+0-00",
    269.0: "0+000+0-0+",
    270.0: "0+000+00-0",
    271.0: "0+000+000-",
    272.0: "0+000+0000",
    273.0: "0+000+000+",
    274.0: "0+000+00+0",
    275.0: "0+000+0+0-",
    276.0: "0+000+0+00",
    277.0: "0+000+0+0+",
    278.0: "0+00+0-0-0",
    279.0: "0+00+0-00-",
    280.0: "0+00+0-000",
    281.0: "0+00+0-00+",
    282.0: "0+00+0-0+0",
    283.0: "0+00+00-0-",
    284.0: "0+00+00-00",
    285.0: "0+00+00-0+",
    286.0: "0+00+000-0",
    287.0: "0+00+0000-",
    288.0: "0+00+00000",
    289.0: "0+00+0000+",
    290.0: "0+00+000+0",
    291.0: "0+00+00+0-",
    292.0: "0+00+00+00",
    293.0: "0+00+00+0+",
    294.0: "0+00+0+0-0",
    295.0: "0+00+0+00-",
    296.0: "0+00+0+000",
    297.0: "0+00+0+00+",
    298.0: "0+00+0+0+0",
    299.0: "0+0+0-0-0-",
    300.0: "0+0+0-0-00",
    301.0: "0+0+0-0-0+",
    302.0: "0+0+0-00-0",
    303.0: "0+0+0-000-",
    304.0: "0+0+0-0000",
    305.0: "0+0+0-000+",
    306.0: "0+0+0-00+0",
    307.0: "0+0+0-0+0-",
    308.0: "0+0+0-0+00",
    309.0: "0+0+0-0+0+",
    310.0: "0+0+00-0-0",
    311.0: "0+0+00-00-",
    312.0: "0+0+00-000",
    313.0: "0+0+00-00+",
    314.0: "0+0+00-0+0",
    315.0: "0+0+000-0-",
    316.0: "0+0+000-00",
    317.0: "0+0+000-0+",
    318.0: "0+0+0000-0",
    319.0: "0+0+00000-",
    320.0: "0+0+000000",
    321.0: "0+0+00000+",
    322.0: "0+0+0000+0",
    323.0: "0+0+000+0-",
    324.0: "0+0+000+00",
    325.0: "0+0+000+0+",
    326.0: "0+0+00+0-0",
    327.0: "0+0+00+00-",
    328.0: "0+0+00+000",
    329.0: "0+0+00+00+",
    330.0: "0+0+00+0+0",
    331.0: "0+0+0+0-0-",
    332.0: "0+0+0+0-00",
    333.0: "0+0+0+0-0+",
    334.0: "0+0+0+00-0",
    335.0: "0+0+0+000-",
    336.0: "0+0+0+0000",
    337.0: "0+0+0+000+",
    338.0: "0+0+0+00+0",
    339.0: "0+0+0+0+0-",
    340.0: "0+0+0+0+00",
    341.0: "0+0+0+0+0+",
    342.0: "+0-0-0-0-0",
    343.0: "+0-0-0-00-",
    344.0: "+0-0-0-000",
    345.0: "+0-0-0-00+",
    346.0: "+0-0-0-0+0",
    347.0: "+0-0-00-0-",
    348.0: "+0-0-00-00",
    349.0: "+0-0-00-0+",
    350.0: "+0-0-000-0",
    351.0: "+0-0-0000-",
    352.0: "+0-0-00000",
    353.0: "+0-0-0000+",
    354.0: "+0-0-000+0",
    355.0: "+0-0-00+0-",
    356.0: "+0-0-00+00",
    357.0: "+0-0-00+0+",
    358.0: "+0-0-0+0-0",
    359.0: "+0-0-0+00-",
    360.0: "+0-0-0+000",
    361.0: "+0-0-0+00+",
    362.0: "+0-0-0+0+0",
    363.0: "+0-00-0-0-",
    364.0: "+0-00-0-00",
    365.0: "+0-00-0-0+",
    366.0: "+0-00-00-0",
    367.0: "+0-00-000-",
    368.0: "+0-00-0000",
    369.0: "+0-00-000+",
    370.0: "+0-00-00+0",
    371.0: "+0-00-0+0-",
    372.0: "+0-00-0+00",
    373.0: "+0-00-0+0+",
    374.0: "+0-000-0-0",
    375.0: "+0-000-00-",
    376.0: "+0-000-000",
    377.0: "+0-000-00+",
    378.0: "+0-000-0+0",
    379.0: "+0-0000-0-",
    380.0: "+0-0000-00",
    381.0: "+0-0000-0+",
    382.0: "+0-00000-0",
    383.0: "+0-000000-",
    384.0: "+0-0000000",
    385.0: "+0-000000+",
    386.0: "+0-00000+0",
    387.0: "+0-0000+0-",
    388.0: "+0-0000+00",
    389.0: "+0-0000+0+",
    390.0: "+0-000+0-0",
    391.0: "+0-000+00-",
    392.0: "+0-000+000",
    393.0: "+0-000+00+",
    394.0: "+0-000+0+0",
    395.0: "+0-00+0-0-",
    396.0: "+0-00+0-00",
    397.0: "+0-00+0-0+",
    398.0: "+0-00+00-0",
    399.0: "+0-00+000-",
    400.0: "+0-00+0000",
    401.0: "+0-00+000+",
    402.0: "+0-00+00+0",
    403.0: "+0-00+0+0-",
    404.0: "+0-00+0+00",
    405.0: "+0-00+0+0+",
    406.0: "+0-0+0-0-0",
    407.0: "+0-0+0-00-",
    408.0: "+0-0+0-000",
    409.0: "+0-0+0-00+",
    410.0: "+0-0+0-0+0",
    411.0: "+0-0+00-0-",
    412.0: "+0-0+00-00",
    413.0: "+0-0+00-0+",
    414.0: "+0-0+000-0",
    415.0: "+0-0+0000-",
    416.0: "+0-0+00000",
    417.0: "+0-0+0000+",
    418.0: "+0-0+000+0",
    419.0: "+0-0+00+0-",
    420.0: "+0-0+00+00",
    421.0: "+0-0+00+0+",
    422.0: "+0-0+0+0-0",
    423.0: "+0-0+0+00-",
    424.0: "+0-0+0+000",
    425.0: "+0-0+0+00+",
    426.0: "+0-0+0+0+0",
    427.0: "+00-0-0-0-",
    428.0: "+00-0-0-00",
    429.0: "+00-0-0-0+",
    430.0: "+00-0-00-0",
    431.0: "+00-0-000-",
    432.0: "+00-0-0000",
    433.0: "+00-0-000+",
    434.0: "+00-0-00+0",
    435.0: "+00-0-0+0-",
    436.0: "+00-0-0+00",
    437.0: "+00-0-0+0+",
    438.0: "+00-00-0-0",
    439.0: "+00-00-00-",
    440.0: "+00-00-000",
    441.0: "+00-00-00+",
    442.0: "+00-00-0+0",
    443.0: "+00-000-0-",
    444.0: "+00-000-00",
    445.0: "+00-000-0+",
    446.0: "+00-0000-0",
    447.0: "+00-00000-",
    448.0: "+00-000000",
    449.0: "+00-00000+",
    450.0: "+00-0000+0",
    451.0: "+00-000+0-",
    452.0: "+00-000+00",
    453.0: "+00-000+0+",
    454.0: "+00-00+0-0",
    455.0: "+00-00+00-",
    456.0: "+00-00+000",
    457.0: "+00-00+00+",
    458.0: "+00-00+0+0",
    459.0: "+00-0+0-0-",
    460.0: "+00-0+0-00",
    461.0: "+00-0+0-0+",
    462.0: "+00-0+00-0",
    463.0: "+00-0+000-",
    464.0: "+00-0+0000",
    465.0: "+00-0+000+",
    466.0: "+00-0+00+0",
    467.0: "+00-0+0+0-",
    468.0: "+00-0+0+00",
    469.0: "+00-0+0+0+",
    470.0: "+000-0-0-0",
    471.0: "+000-0-00-",
    472.0: "+000-0-000",
    473.0: "+000-0-00+",
    474.0: "+000-0-0+0",
    475.0: "+000-00-0-",
    476.0: "+000-00-00",
    477.0: "+000-00-0+",
    478.0: "+000-000-0",
    479.0: "+000-0000-",
    480.0: "+000-00000",
    481.0: "+000-0000+",
    482.0: "+000-000+0",
    483.0: "+000-00+0-",
    484.0: "+000-00+00",
    485.0: "+000-00+0+",
    486.0: "+000-0+0-0",
    487.0: "+000-0+00-",
    488.0: "+000-0+000",
    489.0: "+000-0+00+",
    490.0: "+000-0+0+0",
    491.0: "+0000-0-0-",
    492.0: "+0000-0-00",
    493.0: "+0000-0-0+",
    494.0: "+0000-00-0",
    495.0: "+0000-000-",
    496.0: "+0000-0000",
    497.0: "+0000-000+",
    498.0: "+0000-00+0",
    499.0: "+0000-0+0-",
    500.0: "+0000-0+00",
    501.0: "+0000-0+0+",
    502.0: "+00000-0-0",
    503.0: "+00000-00-",
    504.0: "+00000-000",
    505.0: "+00000-00+",
    506.0: "+00000-0+0",
    507.0: "+000000-0-",
    508.0: "+000000-00",
    509.0: "+000000-0+",
    510.0: "+0000000-0",
    511.0: "+00000000-",
    512.0: "+000000000",
    513.0: "+00000000+",
    514.0: "+0000000+0",
    515.0: "+000000+0-",
    516.0: "+000000+00",
    517.0: "+000000+0+",
    518.0: "+00000+0-0",
    519.0: "+00000+00-",
    520.0: "+00000+000",
    521.0: "+00000+00+",
    522.0: "+00000+0+0",
    523.0: "+0000+0-0-",
    524.0: "+0000+0-00",
    525.0: "+0000+0-0+",
    526.0: "+0000+00-0",
    527.0: "+0000+000-",
    528.0: "+0000+0000",
    529.0: "+0000+000+",
    530.0: "+0000+00+0",
    531.0: "+0000+0+0-",
    532.0: "+0000+0+00",
    533.0: "+0000+0+0+",
    534.0: "+000+0-0-0",
    535.0: "+000+0-00-",
    536.0: "+000+0-000",
    537.0: "+000+0-00+",
    538.0: "+000+0-0+0",
    539.0: "+000+00-0-",
    540.0: "+000+00-00",
    541.0: "+000+00-0+",
    542.0: "+000+000-0",
    543.0: "+000+0000-",
    544.0: "+000+00000",
    545.0: "+000+0000+",
    546.0: "+000+000+0",
    547.0: "+000+00+0-",
    548.0: "+000+00+00",
    549.0: "+000+00+0+",
    550.0: "+000+0+0-0",
    551.0: "+000+0+00-",
    552.0: "+000+0+000",
    553.0: "+000+0+00+",
    554.0: "+000+0+0+0",
    555.0: "+00+0-0-0-",
    556.0: "+00+0-0-00",
    557.0: "+00+0-0-0+",
    558.0: "+00+0-00-0",
    559.0: "+00+0-000-",
    560.0: "+00+0-0000",
    561.0: "+00+0-000+",
    562.0: "+00+0-00+0",
    563.0: "+00+0-0+0-",
    564.0: "+00+0-0+00",
    565.0: "+00+0-0+0+",
    566.0: "+00+00-0-0",
    567.0: "+00+00-00-",
    568.0: "+00+00-000",
    569.0: "+00+00-00+",
    570.0: "+00+00-0+0",
    571.0: "+00+000-0-",
    572.0: "+00+000-00",
    573.0: "+00+000-0+",
    574.0: "+00+0000-0",
    575.0: "+00+00000-",
    576.0: "+00+000000",
    577.0: "+00+00000+",
    578.0: "+00+0000+0",
    579.0: "+00+000+0-",
    580.0: "+00+000+00",
    581.0: "+00+000+0+",
    582.0: "+00+00+0-0",
    583.0: "+00+00+00-",
    584.0: "+00+00+000",
    585.0: "+00+00+00+",
    586.0: "+00+00+0+0",
    587.0: "+00+0+0-0-",
    588.0: "+00+0+0-00",
    589.0: "+00+0+0-0+",
    590.0: "+00+0+00-0",
    591.0: "+00+0+000-",
    592.0: "+00+0+0000",
    593.0: "+00+0+000+",
    594.0: "+00+0+00+0",
    595.0: "+00+0+0+0-",
    596.0: "+00+0+0+00",
    597.0: "+00+0+0+0+",
    598.0: "+0+0-0-0-0",
    599.0: "+0+0-0-00-",
    600.0: "+0+0-0-000",
    601.0: "+0+0-0-00+",
    602.0: "+0+0-0-0+0",
    603.0: "+0+0-00-0-",
    604.0: "+0+0-00-00",
    605.0: "+0+0-00-0+",
    606.0: "+0+0-000-0",
    607.0: "+0+0-0000-",
    608.0: "+0+0-00000",
    609.0: "+0+0-0000+",
    610.0: "+0+0-000+0",
    611.0: "+0+0-00+0-",
    612.0: "+0+0-00+00",
    613.0: "+0+0-00+0+",
    614.0: "+0+0-0+0-0",
    615.0: "+0+0-0+00-",
    616.0: "+0+0-0+000",
    617.0: "+0+0-0+00+",
    618.0: "+0+0-0+0+0",
    619.0: "+0+00-0-0-",
    620.0: "+0+00-0-00",
    621.0: "+0+00-0-0+",
    622.0: "+0+00-00-0",
    623.0: "+0+00-000-",
    624.0: "+0+00-0000",
    625.0: "+0+00-000+",
    626.0: "+0+00-00+0",
    627.0: "+0+00-0+0-",
    628.0: "+0+00-0+00",
    629.0: "+0+00-0+0+",
    630.0: "+0+000-0-0",
    631.0: "+0+000-00-",
    632.0: "+0+000-000",
    633.0: "+0+000-00+",
    634.0: "+0+000-0+0",
    635.0: "+0+0000-0-",
    636.0: "+0+0000-00",
    637.0: "+0+0000-0+",
    638.0: "+0+00000-0",
    639.0: "+0+000000-",
    640.0: "+0+0000000",
    641.0: "+0+000000+",
    642.0: "+0+00000+0",
    643.0: "+0+0000+0-",
    644.0: "+0+0000+00",
    645.0: "+0+0000+0+",
    646.0: "+0+000+0-0",
    647.0: "+0+000+00-",
    648.0: "+0+000+000",
    649.0: "+0+000+00+",
    650.0: "+0+000+0+0",
    651.0: "+0+00+0-0-",
    652.0: "+0+00+0-00",
    653.0: "+0+00+0-0+",
    654.0: "+0+00+00-0",
    655.0: "+0+00+000-",
    656.0: "+0+00+0000",
    657.0: "+0+00+000+",
    658.0: "+0+00+00+0",
    659.0: "+0+00+0+0-",
    660.0: "+0+00+0+00",
    661.0: "+0+00+0+0+",
    662.0: "+0+0+0-0-0",
    663.0: "+0+0+0-00-",
    664.0: "+0+0+0-000",
    665.0: "+0+0+0-00+",
    666.0: "+0+0+0-0+0",
    667.0: "+0+0+00-0-",
    668.0: "+0+0+00-00",
    669.0: "+0+0+00-0+",
    670.0: "+0+0+000-0",
    671.0: "+0+0+0000-",
    672.0: "+0+0+00000",
    673.0: "+0+0+0000+",
    674.0: "+0+0+000+0",
    675.0: "+0+0+00+0-",
    676.0: "+0+0+00+00",
    677.0: "+0+0+00+0+",
    678.0: "+0+0+0+0-0",
    679.0: "+0+0+0+00-",
    680.0: "+0+0+0+000",
    681.0: "+0+0+0+00+",
    682.0: "+0+0+0+0+0",
    -11.0: "00000-0+0+",
    -682.0: "-0-0-0-0-0",
    -681.0: "-0-0-0-00-",
    -680.0: "-0-0-0-000",
    -679.0: "-0-0-0-00+",
    -678.0: "-0-0-0-0+0",
    -677.0: "-0-0-00-0-",
    -676.0: "-0-0-00-00",
    -675.0: "-0-0-00-0+",
    -674.0: "-0-0-000-0",
    -673.0: "-0-0-0000-",
    -672.0: "-0-0-00000",
    -671.0: "-0-0-0000+",
    -670.0: "-0-0-000+0",
    -669.0: "-0-0-00+0-",
    -668.0: "-0-0-00+00",
    -667.0: "-0-0-00+0+",
    -666.0: "-0-0-0+0-0",
    -665.0: "-0-0-0+00-",
    -664.0: "-0-0-0+000",
    -663.0: "-0-0-0+00+",
    -662.0: "-0-0-0+0+0",
    -661.0: "-0-00-0-0-",
    -660.0: "-0-00-0-00",
    -659.0: "-0-00-0-0+",
    -658.0: "-0-00-00-0",
    -657.0: "-0-00-000-",
    -656.0: "-0-00-0000",
    -655.0: "-0-00-000+",
    -654.0: "-0-00-00+0",
    -653.0: "-0-00-0+0-",
    -652.0: "-0-00-0+00",
    -651.0: "-0-00-0+0+",
    -650.0: "-0-000-0-0",
    -649.0: "-0-000-00-",
    -648.0: "-0-000-000",
    -647.0: "-0-000-00+",
    -646.0: "-0-000-0+0",
    -645.0: "-0-0000-0-",
    -644.0: "-0-0000-00",
    -643.0: "-0-0000-0+",
    -642.0: "-0-00000-0",
    -641.0: "-0-000000-",
    -640.0: "-0-0000000",
    -639.0: "-0-000000+",
    -638.0: "-0-00000+0",
    -637.0: "-0-0000+0-",
    -636.0: "-0-0000+00",
    -635.0: "-0-0000+0+",
    -634.0: "-0-000+0-0",
    -633.0: "-0-000+00-",
    -632.0: "-0-000+000",
    -631.0: "-0-000+00+",
    -630.0: "-0-000+0+0",
    -629.0: "-0-00+0-0-",
    -628.0: "-0-00+0-00",
    -627.0: "-0-00+0-0+",
    -626.0: "-0-00+00-0",
    -625.0: "-0-00+000-",
    -624.0: "-0-00+0000",
    -623.0: "-0-00+000+",
    -622.0: "-0-00+00+0",
    -621.0: "-0-00+0+0-",
    -620.0: "-0-00+0+00",
    -619.0: "-0-00+0+0+",
    -618.0: "-0-0+0-0-0",
    -617.0: "-0-0+0-00-",
    -616.0: "-0-0+0-000",
    -615.0: "-0-0+0-00+",
    -614.0: "-0-0+0-0+0",
    -613.0: "-0-0+00-0-",
    -612.0: "-0-0+00-00",
    -611.0: "-0-0+00-0+",
    -610.0: "-0-0+000-0",
    -609.0: "-0-0+0000-",
    -608.0: "-0-0+00000",
    -607.0: "-0-0+0000+",
    -606.0: "-0-0+000+0",
    -605.0: "-0-0+00+0-",
    -604.0: "-0-0+00+00",
    -603.0: "-0-0+00+0+",
    -602.0: "-0-0+0+0-0",
    -601.0: "-0-0+0+00-",
    -600.0: "-0-0+0+000",
    -599.0: "-0-0+0+00+",
    -598.0: "-0-0+0+0+0",
    -597.0: "-00-0-0-0-",
    -596.0: "-00-0-0-00",
    -595.0: "-00-0-0-0+",
    -594.0: "-00-0-00-0",
    -593.0: "-00-0-000-",
    -592.0: "-00-0-0000",
    -591.0: "-00-0-000+",
    -590.0: "-00-0-00+0",
    -589.0: "-00-0-0+0-",
    -588.0: "-00-0-0+00",
    -587.0: "-00-0-0+0+",
    -586.0: "-00-00-0-0",
    -585.0: "-00-00-00-",
    -584.0: "-00-00-000",
    -583.0: "-00-00-00+",
    -582.0: "-00-00-0+0",
    -581.0: "-00-000-0-",
    -580.0: "-00-000-00",
    -579.0: "-00-000-0+",
    -578.0: "-00-0000-0",
    -577.0: "-00-00000-",
    -576.0: "-00-000000",
    -575.0: "-00-00000+",
    -574.0: "-00-0000+0",
    -573.0: "-00-000+0-",
    -572.0: "-00-000+00",
    -571.0: "-00-000+0+",
    -570.0: "-00-00+0-0",
    -569.0: "-00-00+00-",
    -568.0: "-00-00+000",
    -567.0: "-00-00+00+",
    -566.0: "-00-00+0+0",
    -565.0: "-00-0+0-0-",
    -564.0: "-00-0+0-00",
    -563.0: "-00-0+0-0+",
    -562.0: "-00-0+00-0",
    -561.0: "-00-0+000-",
    -560.0: "-00-0+0000",
    -559.0: "-00-0+000+",
    -558.0: "-00-0+00+0",
    -557.0: "-00-0+0+0-",
    -556.0: "-00-0+0+00",
    -555.0: "-00-0+0+0+",
    -554.0: "-000-0-0-0",
    -553.0: "-000-0-00-",
    -552.0: "-000-0-000",
    -551.0: "-000-0-00+",
    -550.0: "-000-0-0+0",
    -549.0: "-000-00-0-",
    -548.0: "-000-00-00",
    -547.0: "-000-00-0+",
    -546.0: "-000-000-0",
    -545.0: "-000-0000-",
    -544.0: "-000-00000",
    -543.0: "-000-0000+",
    -542.0: "-000-000+0",
    -541.0: "-000-00+0-",
    -540.0: "-000-00+00",
    -539.0: "-000-00+0+",
    -538.0: "-000-0+0-0",
    -537.0: "-000-0+00-",
    -536.0: "-000-0+000",
    -535.0: "-000-0+00+",
    -534.0: "-000-0+0+0",
    -533.0: "-0000-0-0-",
    -532.0: "-0000-0-00",
    -531.0: "-0000-0-0+",
    -530.0: "-0000-00-0",
    -529.0: "-0000-000-",
    -528.0: "-0000-0000",
    -527.0: "-0000-000+",
    -526.0: "-0000-00+0",
    -525.0: "-0000-0+0-",
    -524.0: "-0000-0+00",
    -523.0: "-0000-0+0+",
    -522.0: "-00000-0-0",
    -521.0: "-00000-00-",
    -520.0: "-00000-000",
    -519.0: "-00000-00+",
    -518.0: "-00000-0+0",
    -517.0: "-000000-0-",
    -516.0: "-000000-00",
    -515.0: "-000000-0+",
    -514.0: "-0000000-0",
    -513.0: "-00000000-",
    -512.0: "-000000000",
    -511.0: "-00000000+",
    -510.0: "-0000000+0",
    -509.0: "-000000+0-",
    -508.0: "-000000+00",
    -507.0: "-000000+0+",
    -506.0: "-00000+0-0",
    -505.0: "-00000+00-",
    -504.0: "-00000+000",
    -503.0: "-00000+00+",
    -502.0: "-00000+0+0",
    -501.0: "-0000+0-0-",
    -500.0: "-0000+0-00",
    -499.0: "-0000+0-0+",
    -498.0: "-0000+00-0",
    -497.0: "-0000+000-",
    -496.0: "-0000+0000",
    -495.0: "-0000+000+",
    -494.0: "-0000+00+0",
    -493.0: "-0000+0+0-",
    -492.0: "-0000+0+00",
    -491.0: "-0000+0+0+",
    -490.0: "-000+0-0-0",
    -489.0: "-000+0-00-",
    -488.0: "-000+0-000",
    -487.0: "-000+0-00+",
    -486.0: "-000+0-0+0",
    -485.0: "-000+00-0-",
    -484.0: "-000+00-00",
    -483.0: "-000+00-0+",
    -482.0: "-000+000-0",
    -481.0: "-000+0000-",
    -480.0: "-000+00000",
    -479.0: "-000+0000+",
    -478.0: "-000+000+0",
    -477.0: "-000+00+0-",
    -476.0: "-000+00+00",
    -475.0: "-000+00+0+",
    -474.0: "-000+0+0-0",
    -473.0: "-000+0+00-",
    -472.0: "-000+0+000",
    -471.0: "-000+0+00+",
    -470.0: "-000+0+0+0",
    -469.0: "-00+0-0-0-",
    -468.0: "-00+0-0-00",
    -467.0: "-00+0-0-0+",
    -466.0: "-00+0-00-0",
    -465.0: "-00+0-000-",
    -464.0: "-00+0-0000",
    -463.0: "-00+0-000+",
    -462.0: "-00+0-00+0",
    -461.0: "-00+0-0+0-",
    -460.0: "-00+0-0+00",
    -459.0: "-00+0-0+0+",
    -458.0: "-00+00-0-0",
    -457.0: "-00+00-00-",
    -456.0: "-00+00-000",
    -455.0: "-00+00-00+",
    -454.0: "-00+00-0+0",
    -453.0: "-00+000-0-",
    -452.0: "-00+000-00",
    -451.0: "-00+000-0+",
    -450.0: "-00+0000-0",
    -449.0: "-00+00000-",
    -448.0: "-00+000000",
    -447.0: "-00+00000+",
    -446.0: "-00+0000+0",
    -445.0: "-00+000+0-",
    -444.0: "-00+000+00",
    -443.0: "-00+000+0+",
    -442.0: "-00+00+0-0",
    -441.0: "-00+00+00-",
    -440.0: "-00+00+000",
    -439.0: "-00+00+00+",
    -438.0: "-00+00+0+0",
    -437.0: "-00+0+0-0-",
    -436.0: "-00+0+0-00",
    -435.0: "-00+0+0-0+",
    -434.0: "-00+0+00-0",
    -433.0: "-00+0+000-",
    -432.0: "-00+0+0000",
    -431.0: "-00+0+000+",
    -430.0: "-00+0+00+0",
    -429.0: "-00+0+0+0-",
    -428.0: "-00+0+0+00",
    -427.0: "-00+0+0+0+",
    -426.0: "-0+0-0-0-0",
    -425.0: "-0+0-0-00-",
    -424.0: "-0+0-0-000",
    -423.0: "-0+0-0-00+",
    -422.0: "-0+0-0-0+0",
    -421.0: "-0+0-00-0-",
    -420.0: "-0+0-00-00",
    -419.0: "-0+0-00-0+",
    -418.0: "-0+0-000-0",
    -417.0: "-0+0-0000-",
    -416.0: "-0+0-00000",
    -415.0: "-0+0-0000+",
    -414.0: "-0+0-000+0",
    -413.0: "-0+0-00+0-",
    -412.0: "-0+0-00+00",
    -411.0: "-0+0-00+0+",
    -410.0: "-0+0-0+0-0",
    -409.0: "-0+0-0+00-",
    -408.0: "-0+0-0+000",
    -407.0: "-0+0-0+00+",
    -406.0: "-0+0-0+0+0",
    -405.0: "-0+00-0-0-",
    -404.0: "-0+00-0-00",
    -403.0: "-0+00-0-0+",
    -402.0: "-0+00-00-0",
    -401.0: "-0+00-000-",
    -400.0: "-0+00-0000",
    -399.0: "-0+00-000+",
    -398.0: "-0+00-00+0",
    -397.0: "-0+00-0+0-",
    -396.0: "-0+00-0+00",
    -395.0: "-0+00-0+0+",
    -394.0: "-0+000-0-0",
    -393.0: "-0+000-00-",
    -392.0: "-0+000-000",
    -391.0: "-0+000-00+",
    -390.0: "-0+000-0+0",
    -389.0: "-0+0000-0-",
    -388.0: "-0+0000-00",
    -387.0: "-0+0000-0+",
    -386.0: "-0+00000-0",
    -385.0: "-0+000000-",
    -384.0: "-0+0000000",
    -383.0: "-0+000000+",
    -382.0: "-0+00000+0",
    -381.0: "-0+0000+0-",
    -380.0: "-0+0000+00",
    -379.0: "-0+0000+0+",
    -378.0: "-0+000+0-0",
    -377.0: "-0+000+00-",
    -376.0: "-0+000+000",
    -375.0: "-0+000+00+",
    -374.0: "-0+000+0+0",
    -373.0: "-0+00+0-0-",
    -372.0: "-0+00+0-00",
    -371.0: "-0+00+0-0+",
    -370.0: "-0+00+00-0",
    -369.0: "-0+00+000-",
    -368.0: "-0+00+0000",
    -367.0: "-0+00+000+",
    -366.0: "-0+00+00+0",
    -365.0: "-0+00+0+0-",
    -364.0: "-0+00+0+00",
    -363.0: "-0+00+0+0+",
    -362.0: "-0+0+0-0-0",
    -361.0: "-0+0+0-00-",
    -360.0: "-0+0+0-000",
    -359.0: "-0+0+0-00+",
    -358.0: "-0+0+0-0+0",
    -357.0: "-0+0+00-0-",
    -356.0: "-0+0+00-00",
    -355.0: "-0+0+00-0+",
    -354.0: "-0+0+000-0",
    -353.0: "-0+0+0000-",
    -352.0: "-0+0+00000",
    -351.0: "-0+0+0000+",
    -350.0: "-0+0+000+0",
    -349.0: "-0+0+00+0-",
    -348.0: "-0+0+00+00",
    -347.0: "-0+0+00+0+",
    -346.0: "-0+0+0+0-0",
    -345.0: "-0+0+0+00-",
    -344.0: "-0+0+0+000",
    -343.0: "-0+0+0+00+",
    -342.0: "-0+0+0+0+0",
    -341.0: "0-0-0-0-0-",
    -340.0: "0-0-0-0-00",
    -339.0: "0-0-0-0-0+",
    -338.0: "0-0-0-00-0",
    -337.0: "0-0-0-000-",
    -336.0: "0-0-0-0000",
    -335.0: "0-0-0-000+",
    -334.0: "0-0-0-00+0",
    -333.0: "0-0-0-0+0-",
    -332.0: "0-0-0-0+00",
    -331.0: "0-0-0-0+0+",
    -330.0: "0-0-00-0-0",
    -329.0: "0-0-00-00-",
    -328.0: "0-0-00-000",
    -327.0: "0-0-00-00+",
    -326.0: "0-0-00-0+0",
    -325.0: "0-0-000-0-",
    -324.0: "0-0-000-00",
    -323.0: "0-0-000-0+",
    -322.0: "0-0-0000-0",
    -321.0: "0-0-00000-",
    -320.0: "0-0-000000",
    -319.0: "0-0-00000+",
    -318.0: "0-0-0000+0",
    -317.0: "0-0-000+0-",
    -316.0: "0-0-000+00",
    -315.0: "0-0-000+0+",
    -314.0: "0-0-00+0-0",
    -313.0: "0-0-00+00-",
    -312.0: "0-0-00+000",
    -311.0: "0-0-00+00+",
    -310.0: "0-0-00+0+0",
    -309.0: "0-0-0+0-0-",
    -308.0: "0-0-0+0-00",
    -307.0: "0-0-0+0-0+",
    -306.0: "0-0-0+00-0",
    -305.0: "0-0-0+000-",
    -304.0: "0-0-0+0000",
    -303.0: "0-0-0+000+",
    -302.0: "0-0-0+00+0",
    -301.0: "0-0-0+0+0-",
    -300.0: "0-0-0+0+00",
    -299.0: "0-0-0+0+0+",
    -298.0: "0-00-0-0-0",
    -297.0: "0-00-0-00-",
    -296.0: "0-00-0-000",
    -295.0: "0-00-0-00+",
    -294.0: "0-00-0-0+0",
    -293.0: "0-00-00-0-",
    -292.0: "0-00-00-00",
    -291.0: "0-00-00-0+",
    -290.0: "0-00-000-0",
    -289.0: "0-00-0000-",
    -288.0: "0-00-00000",
    -287.0: "0-00-0000+",
    -286.0: "0-00-000+0",
    -285.0: "0-00-00+0-",
    -284.0: "0-00-00+00",
    -283.0: "0-00-00+0+",
    -282.0: "0-00-0+0-0",
    -281.0: "0-00-0+00-",
    -280.0: "0-00-0+000",
    -279.0: "0-00-0+00+",
    -278.0: "0-00-0+0+0",
    -277.0: "0-000-0-0-",
    -276.0: "0-000-0-00",
    -275.0: "0-000-0-0+",
    -274.0: "0-000-00-0",
    -273.0: "0-000-000-",
    -272.0: "0-000-0000",
    -271.0: "0-000-000+",
    -270.0: "0-000-00+0",
    -269.0: "0-000-0+0-",
    -268.0: "0-000-0+00",
    -267.0: "0-000-0+0+",
    -266.0: "0-0000-0-0",
    -265.0: "0-0000-00-",
    -264.0: "0-0000-000",
    -263.0: "0-0000-00+",
    -262.0: "0-0000-0+0",
    -261.0: "0-00000-0-",
    -260.0: "0-00000-00",
    -259.0: "0-00000-0+",
    -258.0: "0-000000-0",
    -257.0: "0-0000000-",
    -256.0: "0-00000000",
    -255.0: "0-0000000+",
    -254.0: "0-000000+0",
    -253.0: "0-00000+0-",
    -252.0: "0-00000+00",
    -251.0: "0-00000+0+",
    -250.0: "0-0000+0-0",
    -249.0: "0-0000+00-",
    -248.0: "0-0000+000",
    -247.0: "0-0000+00+",
    -246.0: "0-0000+0+0",
    -245.0: "0-000+0-0-",
    -244.0: "0-000+0-00",
    -243.0: "0-000+0-0+",
    -242.0: "0-000+00-0",
    -241.0: "0-000+000-",
    -240.0: "0-000+0000",
    -239.0: "0-000+000+",
    -238.0: "0-000+00+0",
    -237.0: "0-000+0+0-",
    -236.0: "0-000+0+00",
    -235.0: "0-000+0+0+",
    -234.0: "0-00+0-0-0",
    -233.0: "0-00+0-00-",
    -232.0: "0-00+0-000",
    -231.0: "0-00+0-00+",
    -230.0: "0-00+0-0+0",
    -229.0: "0-00+00-0-",
    -228.0: "0-00+00-00",
    -227.0: "0-00+00-0+",
    -226.0: "0-00+000-0",
    -225.0: "0-00+0000-",
    -224.0: "0-00+00000",
    -223.0: "0-00+0000+",
    -222.0: "0-00+000+0",
    -221.0: "0-00+00+0-",
    -220.0: "0-00+00+00",
    -219.0: "0-00+00+0+",
    -218.0: "0-00+0+0-0",
    -217.0: "0-00+0+00-",
    -216.0: "0-00+0+000",
    -215.0: "0-00+0+00+",
    -214.0: "0-00+0+0+0",
    -213.0: "0-0+0-0-0-",
    -212.0: "0-0+0-0-00",
    -211.0: "0-0+0-0-0+",
    -210.0: "0-0+0-00-0",
    -209.0: "0-0+0-000-",
    -208.0: "0-0+0-0000",
    -207.0: "0-0+0-000+",
    -206.0: "0-0+0-00+0",
    -205.0: "0-0+0-0+0-",
    -204.0: "0-0+0-0+00",
    -203.0: "0-0+0-0+0+",
    -202.0: "0-0+00-0-0",
    -201.0: "0-0+00-00-",
    -200.0: "0-0+00-000",
    -199.0: "0-0+00-00+",
    -198.0: "0-0+00-0+0",
    -197.0: "0-0+000-0-",
    -196.0: "0-0+000-00",
    -195.0: "0-0+000-0+",
    -194.0: "0-0+0000-0",
    -193.0: "0-0+00000-",
    -192.0: "0-0+000000",
    -191.0: "0-0+00000+",
    -190.0: "0-0+0000+0",
    -189.0: "0-0+000+0-",
    -188.0: "0-0+000+00",
    -187.0: "0-0+000+0+",
    -186.0: "0-0+00+0-0",
    -185.0: "0-0+00+00-",
    -184.0: "0-0+00+000",
    -183.0: "0-0+00+00+",
    -182.0: "0-0+00+0+0",
    -181.0: "0-0+0+0-0-",
    -180.0: "0-0+0+0-00",
    -179.0: "0-0+0+0-0+",
    -178.0: "0-0+0+00-0",
    -177.0: "0-0+0+000-",
    -176.0: "0-0+0+0000",
    -175.0: "0-0+0+000+",
    -174.0: "0-0+0+00+0",
    -173.0: "0-0+0+0+0-",
    -172.0: "0-0+0+0+00",
    -171.0: "0-0+0+0+0+",
    -170.0: "00-0-0-0-0",
    -169.0: "00-0-0-00-",
    -168.0: "00-0-0-000",
    -167.0: "00-0-0-00+",
    -166.0: "00-0-0-0+0",
    -165.0: "00-0-00-0-",
    -164.0: "00-0-00-00",
    -163.0: "00-0-00-0+",
    -162.0: "00-0-000-0",
    -161.0: "00-0-0000-",
    -160.0: "00-0-00000",
    -159.0: "00-0-0000+",
    -158.0: "00-0-000+0",
    -157.0: "00-0-00+0-",
    -156.0: "00-0-00+00",
    -155.0: "00-0-00+0+",
    -154.0: "00-0-0+0-0",
    -153.0: "00-0-0+00-",
    -152.0: "00-0-0+000",
    -151.0: "00-0-0+00+",
    -150.0: "00-0-0+0+0",
    -149.0: "00-00-0-0-",
    -148.0: "00-00-0-00",
    -147.0: "00-00-0-0+",
    -146.0: "00-00-00-0",
    -145.0: "00-00-000-",
    -144.0: "00-00-0000",
    -143.0: "00-00-000+",
    -142.0: "00-00-00+0",
    -141.0: "00-00-0+0-",
    -140.0: "00-00-0+00",
    -139.0: "00-00-0+0+",
    -138.0: "00-000-0-0",
    -137.0: "00-000-00-",
    -136.0: "00-000-000",
    -135.0: "00-000-00+",
    -134.0: "00-000-0+0",
    -133.0: "00-0000-0-",
    -132.0: "00-0000-00",
    -131.0: "00-0000-0+",
    -130.0: "00-00000-0",
    -129.0: "00-000000-",
    -128.0: "00-0000000",
    -127.0: "00-000000+",
    -126.0: "00-00000+0",
    -125.0: "00-0000+0-",
    -124.0: "00-0000+00",
    -123.0: "00-0000+0+",
    -122.0: "00-000+0-0",
    -121.0: "00-000+00-",
    -120.0: "00-000+000",
    -119.0: "00-000+00+",
    -118.0: "00-000+0+0",
    -117.0: "00-00+0-0-",
    -116.0: "00-00+0-00",
    -115.0: "00-00+0-0+",
    -114.0: "00-00+00-0",
    -113.0: "00-00+000-",
    -112.0: "00-00+0000",
    -111.0: "00-00+000+",
    -110.0: "00-00+00+0",
    -109.0: "00-00+0+0-",
    -108.0: "00-00+0+00",
    -107.0: "00-00+0+0+",
    -106.0: "00-0+0-0-0",
    -105.0: "00-0+0-00-",
    -104.0: "00-0+0-000",
    -103.0: "00-0+0-00+",
    -102.0: "00-0+0-0+0",
    -101.0: "00-0+00-0-",
    -100.0: "00-0+00-00",
    -99.0: "00-0+00-0+",
    -98.0: "00-0+000-0",
    -97.0: "00-0+0000-",
    -96.0: "00-0+00000",
    -95.0: "00-0+0000+",
    -94.0: "00-0+000+0",
    -93.0: "00-0+00+0-",
    -92.0: "00-0+00+00",
    -91.0: "00-0+00+0+",
    -90.0: "00-0+0+0-0",
    -89.0: "00-0+0+00-",
    -88.0: "00-0+0+000",
    -87.0: "00-0+0+00+",
    -86.0: "00-0+0+0+0",
    -85.0: "000-0-0-0-",
    -84.0: "000-0-0-00",
    -83.0: "000-0-0-0+",
    -82.0: "000-0-00-0",
    -81.0: "000-0-000-",
    -80.0: "000-0-0000",
    -79.0: "000-0-000+",
    -78.0: "000-0-00+0",
    -77.0: "000-0-0+0-",
    -76.0: "000-0-0+00",
    -75.0: "000-0-0+0+",
    -74.0: "000-00-0-0",
    -73.0: "000-00-00-",
    -72.0: "000-00-000",
    -71.0: "000-00-00+",
    -70.0: "000-00-0+0",
    -69.0: "000-000-0-",
    -68.0: "000-000-00",
    -67.0: "000-000-0+",
    -66.0: "000-0000-0",
    -65.0: "000-00000-",
    -64.0: "000-000000",
    -63.0: "000-00000+",
    -62.0: "000-0000+0",
    -61.0: "000-000+0-",
    -60.0: "000-000+00",
    -59.0: "000-000+0+",
    -58.0: "000-00+0-0",
    -57.0: "000-00+00-",
    -56.0: "000-00+000",
    -55.0: "000-00+00+",
    -54.0: "000-00+0+0",
    -53.0: "000-0+0-0-",
    -52.0: "000-0+0-00",
    -51.0: "000-0+0-0+",
    -50.0: "000-0+00-0",
    -49.0: "000-0+000-",
    -48.0: "000-0+0000",
    -47.0: "000-0+000+",
    -46.0: "000-0+00+0",
    -45.0: "000-0+0+0-",
    -44.0: "000-0+0+00",
    -43.0: "000-0+0+0+",
    -42.0: "0000-0-0-0",
    -41.0: "0000-0-00-",
    -40.0: "0000-0-000",
    -39.0: "0000-0-00+",
    -38.0: "0000-0-0+0",
    -37.0: "0000-00-0-",
    -36.0: "0000-00-00",
    -35.0: "0000-00-0+",
    -34.0: "0000-000-0",
    -33.0: "0000-0000-",
    -32.0: "0000-00000",
    -31.0: "0000-0000+",
    -30.0: "0000-000+0",
    -29.0: "0000-00+0-",
    -28.0: "0000-00+00",
    -27.0: "0000-00+0+",
    -26.0: "0000-0+0-0",
    -25.0: "0000-0+00-",
    -24.0: "0000-0+000",
    -23.0: "0000-0+00+",
    -22.0: "0000-0+0+0",
    -21.0: "00000-0-0-",
    -20.0: "00000-0-00",
    -19.0: "00000-0-0+",
    -18.0: "00000-00-0",
    -17.0: "00000-000-",
    -16.0: "00000-0000",
    -15.0: "00000-000+",
    -14.0: "00000-00+0",
    -13.0: "00000-0+0-",
    -12.0: "00000-0+00",
    -1.0: "000000000-",
    -10.0: "000000-0-0",
    -9.0: "000000-00-",
    -8.0: "000000-000",
    -7.0: "000000-00+",
    -6.0: "000000-0+0",
    -5.0: "0000000-0-",
    -4.0: "0000000-00",
    -3.0: "0000000-0+",
    -2.0: "00000000-0",
}


class tests__to_and_fro_10bits(unittest.TestCase):
    def test__01_to_integer(self):
        """Check conversion from CSD to integer"""

        for key in good_values_dict.keys():
            csd_str = good_values_dict[key]
            value = csd.to_decimal(csd_str)
            self.assertEqual(value, key)

    def test__02_to_csd(self):
        """Check that integers are converted to CSD properly."""

        for key in good_values_dict.keys():
            csd_str = csd.to_csd(key)

            while len(csd_str) < 10:
                csd_str = "0" + csd_str

            self.assertEqual(csd_str, good_values_dict[key])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(tests__to_and_fro_10bits))
    return suite


if __name__ == "__main__":
    unittest.main()
