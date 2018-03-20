from orangecontrib.associate.fpgrowth import * 


data = [['A','B','C'        ],
		[    'B',    'D','E'],
		['A','B',    'D','E'],
		['A'                ],
		['A','B','C','D','E'],
		['A',            'E'],
		['A',    'C'        ]]


itemsets = list(frequent_itemsets(data,0.4))
print('\n\nFrequent Items:---------\n\n')
print(*itemsets, sep="\n")

itemsets = dict(frequent_itemsets(data, .4))
rules =list(association_rules(itemsets, .9))
print('\n\nAsscoiation Rule:\n\n')
print(*rules, sep="\n")