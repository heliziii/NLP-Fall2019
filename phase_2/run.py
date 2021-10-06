import english
lang = int(input("Please enter the language (enter 1 or 2):\n1)English\n2)Persian\n"))
address = input("Phase 1: Please enter the document address\n")
if lang == 1:
    dic, most_occur = english.phase1(address)
    print('Dictionary:')
    print(dic)
    print('Most Occur:')
    print(most_occur)
    while True:
        query = input('Phase 2: Enter query\n')
        if query == 'next':
            break
        print(english.posting_list(query))
    #english.phase3()
    while True:
        query = input('Phase 4: Enter query\n')
        if query == 'next':
            break
        print(english.query_edit(query))
    wtd, tf = english.wtd_compute()
    class_prediction = english.prediction()
    while True:
        query = input('Phase 5 type I search: Enter query\n')
        class_number = int(input('Phase 5 type I search: Enter class\n'))
        if query == 'next':
            break
        print(english.query_search_vector(query,wtd, tf, class_prediction, class_number))
    while True:
        query = input('Phase 5 type II search: Enter query\n')
        if query == 'next':
            break
        window = int(input('Phase 5 type II search: Enter window length\n'))
        print(english.query_search_proxmity(query, window))

