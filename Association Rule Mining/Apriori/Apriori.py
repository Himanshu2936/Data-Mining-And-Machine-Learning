import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = [['A','B','C'        ],
		[    'B',    'D','E'],
		['A','B',    'D','E'],
		['A'                ],
		['A','B','C','D','E'],
		['A',            'E'],
		['A',    'C'        ]]
		
oht = OnehotTransactions()
oht_ary = oht.fit(dataset).transform(dataset)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

print(frequent_itemsets)
rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
print(rules.iloc[:,0:4])