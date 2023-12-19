from Functions import *


class CBR:
   
    def __init__(self,case_base=DecisionTree()):
        """
        Initializes a CBR instance with optional case base.
            :param case_base: The decision tree to be used for case-based reasoning, defaults to a new DecisionTree instance.
        """
        # Reading CSV files
        self.books=pd.read_csv("../ontologia/Books.csv",low_memory=False)
        self.cases=pd.read_csv("../ontologia/Cases.csv")
        self.users=pd.read_csv("../ontologia/Users.csv")

         # Define ID for new cases to be added to CSV, keeping track of all cases in the database
         self.id_case_1 = len(self.cases)


        # Convert data to appropriate dtypes
        self.books['traduccions'] = self.books.traduccions.apply(literal_eval)
        self.books['pertany_a'] = self.books.pertany_a.apply(literal_eval)
        self.books['edat_minima'] = self.books['edat_minima'].astype(int)
        self.books['num_pagines'] = self.books['num_pagines'].astype(int)
        self.books['any_publicacio'] = self.books['any_publicacio'].astype(int)
        self.cases['Idioma'] = self.cases.Idioma.apply(literal_eval)

        # Important variables
        self.range_pages_books = self.books['num_pagines'].max()-self.books['num_pagines'].min()
        self.any_publi_range = self.books['any_publicacio'].max()-self.books['any_publicacio'].min()
        self.edat_minima_range = self.books['edat_minima'].max()-self.books['edat_minima'].min()
        self.range_year = self.cases['any_naixement'].max()-self.cases['any_naixement'].min()
        self.range_pages = self.cases['pagines_max'].max()-self.cases['pagines_max'].min()
        self.features_by_importance = ['genere_persona', 'pref_sagues', 'pref_tipus_lectura', 'Romàntic', 'Ciència Ficció', 'any_naixement', 
                                  'Comèdia', 'Històrica', 'Ficció', 'Fantasia', 'Ciència', 'Creixement personal', 'Policiaca', 'Juvenil', 
                                  'pagines_max', 'pref_adaptacio_peli', 'pref_best_seller']


        # Case Tree
        self.case_base = case_base

        # Define attribute variables
        self.genre_options = ['Romàntic','Ciència Ficció', 'Comèdia', 'Històrica', 'Ficció', 'Fantasia', 'Ciència', 'Creixement personal', 'Policiaca', 'Juvenil']
        self.language_options = ['Alemany', 'Albanès', 'Anglès', 'Búlgar', 'Català', 'Coreà', 'Croat', 'Danès', 'Espanyol', 'Finès', 'Francès', 'Gallec', 'Grec', 'Hebreu', 'Hongarès', 'Italià', 'Japonès', 'Letó', 'Neerlandès', 'Noruec', 'Polonès', 'Portuguès', 'Romanès', 'Rus', 'Serbi', 'Suec', 'Tailandès', 'Turc', 'Txec', 'Xinès']
        self.sex_options=['Dona','Home','Altres']
        self.film_options=['Si','No','Indiferent']
        self.bestseller_options=['Si','No','Indiferent']
        self.saga_options=['Si','No','Indiferent']
        self.reading_options=['Densa','Fluida','Indiferent']
        self.title = 'Recomanador de Llibres 📚'



    ## GETTERS 
    def get_genre_options(self):
        """
        Returns the available genre options.
            :return: A list of genre options.
        """
        return self.genre_options
    def get_language_options(self):
        """
        Returns the available language options.
            :return: A list of language options.
        """
        return self.language_options
    def get_sex_options(self):
        """
        Returns the available gender options.
            :return: A list of gender options.
        """
        return self.sex_options
    def get_film_options(self):
        """
        Returns the available film adaptation preference options.
            :return: A list of film adaptation preference options.
        """
        return self.film_options
    def get_bestseller_options(self):
        """
        Returns the available bestseller preference options.
            :return: A list of bestseller preference options.
        """
        return self.bestseller_options
    def get_saga_options(self):
        """
        Returns the available saga preference options.
            :return: A list of saga preference options.
        """
        return self.saga_options
    def get_reading_options(self):
        """
        Returns the available reading type options.
            :return: A list of reading type options.
        """
        return self.reading_options
    
    def last_user(self):
        """
        Retrieves the ID of the last user in the system.
            :return: The ID of the last user.
        """
        return len(self.users)
    
    def last_case(self):
        """
        Retrieves the ID of the last case in the system.
            :return: The ID of the last case.
        """
        return len(self.cases)
    
    def add_user(self):
        """
        Adds a new, empty user to the system.
        """
        self.users.loc[self.last_user()] = [self.last_user()+1,'']
    
    def delete_last_if_empty(self):
        """
        Deletes the last user if they have not been filled with any data.
        """
        if len(self.users.loc[self.last_user()-1]['nom']) == 0:
            self.users.drop(self.last_user()-1,inplace=True)

    def user_data(self, id):
        """
        Retrieves the data of the user currently using the recommender.
        Sets default values for new users, and retrieves the last preferences for existing users.
            :param id: The unique identifier (integer) of the user.
            :return: A tuple containing the name of the user (string) and a dictionary of their data (dict).
        """
        # Finds the user in the DataFrame by their user ID and gets their information.
        usuari = self.users.loc[self.users['id_usuari'] == id].iloc[0]
        nom=usuari[1]
        data = {}

        try:
             # Attempt to retrieve the most recent record for the user in the 'cases' DataFrame.
            aux = self.cases[self.cases["id_usuari"] == id].tail(1).to_dict(orient='records')[0]

            # Extracts and stores various user preferences from the last time they used the recommender.
            data['any_naixement'] = aux['any_naixement']
            data['genere_persona'] = self.sex_options.index(aux['genere_persona'])
            for col in self.genre_options:
                data[col] = aux[col]
            data['idioma'] = aux['Idioma']
            data['pref_adaptacio_peli'] = self.film_options.index(aux['pref_adaptacio_peli'])
            data['pref_best_seller'] = self.bestseller_options.index(aux['pref_best_seller'])
            data['pref_sagues'] = self.saga_options.index(aux['pref_sagues'])
            data['pref_tipus_lectura'] = self.reading_options.index(aux['pref_tipus_lectura'])
            data['pagines_max'] = aux['pagines_max']
            
        # If the user is new, set default values for their preferences.
        except:
            data['any_naixement'] = 2000
            data['genere_persona'] = 0
            data['pref_adaptacio_peli'] = 2
            data['pref_best_seller'] = 2
            data['pref_sagues'] = 2
            data['pref_tipus_lectura'] = 2
            data['pagines_max'] = 1000
            for col in self.genre_options:
                data[col] = 0
            data['idioma']=[]
        
        data['id_usuari'] = id
    
        return nom, data
   
    def read(self,id):
        """
        Retrieves the books that the user has read.
            :param id: The unique identifier (integer) of the user.
            :return: A tuple containing:
                     - llegits (DataFrame): DataFrame containing details of the books read by the user.
                     - camps_no_editables (tuple): A tuple of column names that are not editable.
                     - configuracio (dict): A dictionary configuring the display names of the columns in 'llegits'.
        """
        df = self.cases[self.cases['id_usuari'] == id]
        llegits = pd.merge(df[['id_usuari','score','id_llibre','comprat']],self.books[['id_llibre','titol' ,'escrit_per']], on='id_llibre')
        # Per ordenar les columnes
        llegits = llegits[['titol', 'escrit_per', 'score','comprat','id_llibre','id_usuari']]
        camps_no_editables =('titol','escrit_per','comprat')   
        # Diccionari amb confirguracio de les columnes: None=no surt, Nom=surt amb aquest nom
        configuracio ={'id_usuari':[None], 'id_llibre':[None],'titol':['Titol'],'escrit_per':['Autor'],'score':['Puntuacio'],'comprat':['Comprat']}
        return llegits,camps_no_editables,configuracio
   
    # Re-guardem el nom si l'usuari el modifica
    def change_user_name(self,id,nom):
        """
        Updates the name of the user if the user modifies it.
            :param id: The unique identifier (integer) of the user.
            :param nom: The new name (string) to be updated for the user.
            :return: None. Modifies the 'nom' field for the user in the users DataFrame.
        """
        self.users.loc[self.users["id_usuari"]==id,'nom'] = nom
        #self.users.to_csv('Users.csv', encoding='utf-8', index=False)
    
    def actualitzar_puntuacions(self,modificats):
        """
        Updates the scores of the books for the user.
            :param modificats: A DataFrame containing the modified scores along with user and book IDs.
            :return: None. Modifies the 'score' field in the 'cases' DataFrame for each user and book combination.
        """
        for idx,fila in modificats.iterrows():
            self.cases.loc[(self.cases['id_usuari']==fila['id_usuari']) & (self.cases['id_llibre']==fila['id_llibre']),'score']=fila['score']
    
    def recomanacions(self,user_id,data):
        """
        Generates book recommendations based on user preferences and data provided.
            :param user_id: The unique identifier (integer) of the user for whom recommendations are being made.
            :param data: A dictionary containing the user's data and preferences.

            :return: A list of tuples, each tuple containing detailed information about a recommended book and the reasoning behind the recommendation.
        """
        data1 = data.copy()
        data1['any_naixement'] = '< 2003' if (data1['any_naixement'] < 2003) else '>= 2003'
        
        self.retrieve(data1)
        self.reuse(data)
        self.adapted = []; self.adapted_solutions = []
        for i in range(len(self.reused)):
            book = self.reused[i]
            solucio = self.revise(book,i)
            self.adapted_solutions.append(solucio)
            case = self.case_problem.copy()
            case.set_solucio(solucio)
            self.adapted.append(case)

        df = self.books[self.books['id_llibre'] == self.adapted[0].solucio]
        df = df.append(self.books[self.books['id_llibre'] == self.adapted[1].solucio])
        df = df.append(self.books[self.books['id_llibre'] == self.adapted[2].solucio])

        # TRAÇA
        trace_data = self.get_trace_data()
        llista=[]
        i=0
        for idx,fila in df.iterrows():
            lectors = fila['num_lectures']
            punts = fila['valoracio']
            
            del trace_data[i]['Case_description']['Comèdia']
            del trace_data[i]['Case_description']['Ficció']
            del trace_data[i]['Case_description']['Ciència Ficció']
            del trace_data[i]['Case_description']['Romàntic']
            del trace_data[i]['Case_description']['Fantasia']
            del trace_data[i]['Case_description']['Ciència']
            del trace_data[i]['Case_description']['Creixement personal']
            del trace_data[i]['Case_description']['Policiaca']
            del trace_data[i]['Case_description']['Juvenil']
            del trace_data[i]['Case_description']['Històrica']

            trace_data[i]['Case_description']['solucio'] = trace_data[i]['Old_solution_name']
            trace_data[i]['Case_description']['generes_list'] = str(trace_data[i]['Case_description']['generes_list'])
            trace_data[i]['Case_description']['idioma'] = str(trace_data[i]['Case_description']['idioma'])
            descripcio = pd.DataFrame(trace_data[i]['Case_description'],index=['Retrieved Case'])
            descripcio_trans = descripcio.T

            trace_data[i]['Old_solution_description']['pertany_a'] = str(trace_data[i]['Old_solution_description']['pertany_a'])
            trace_data[i]['Old_solution_description']['format'] = str(trace_data[i]['Old_solution_description']['format'])
            trace_data[i]['Old_solution_description']['traduccions'] = str(trace_data[i]['Old_solution_description']['traduccions'])
            Old_solution_df = pd.DataFrame(trace_data[i]['Old_solution_description'],index=['Retrieved Solution'])
            Old_solution_trans = Old_solution_df.T

            trace_1 = "Basant-nos en les teves preferències, hem trobat que el cas més similar al teu i que, alhora, més va agradar la recomanació que vam donar va ser:"
            trace_2 = f"Amb una similitud entre els dos casos de {trace_data[i]['Similarity_case']:.2f}"
            
            if trace_data[i]['New_solution']is None:
                trace_3 = "Donat que la solució que es va donar al cas més similar, ja compleix amb totes les teves restriccions hem decidit recomanar-te'l:"
                llista.append((fila['titol'],fila['escrit_per'],trace_1,fila['id_llibre'],user_id,trace_2,trace_3,descripcio_trans,Old_solution_trans))
            
            else:
                trace_data[i]['New_solution_description']['pertany_a'] = str(trace_data[i]['New_solution_description']['pertany_a'])
                trace_data[i]['New_solution_description']['format'] = str(trace_data[i]['New_solution_description']['format'])
                trace_data[i]['New_solution_description']['traduccions'] = str(trace_data[i]['New_solution_description']['traduccions'])
                New_solution_df = pd.DataFrame(trace_data[i]['New_solution_description'],index=['Adapted Solution'])
                New_solution_trans = New_solution_df.T
                trace_3 = f"A priori t'anavem a recomanar {trace_data[i]['Old_solution_name']} però no complia les resticcions: {trace_data[i]['Broken_restrictions']}. Tenint en compte això hem trobat el llibre que més se li assembla, amb una similitud entre llibres de {trace_data[i]['Similarity_book']:.3f} i que compleix les teves preferencies:"
                con_solutions = pd.concat([Old_solution_trans,New_solution_trans],axis=1)
                llista.append((fila['titol'],fila['escrit_per'],trace_1,fila['id_llibre'],user_id,trace_2,trace_3,descripcio_trans,con_solutions))
            i+=1
        return llista
    
    def similarity(self,v1,v2,features_ordered, cases=True, retain = False):
        """
        Calculates the similarity between two vectors, v1 and v2, based on a set of features.
            :param v1: The reference vector (or case) from which the similarity is computed.
            :param v2: The vector to compare against the reference vector.
            :param features_ordered: A list of features in the order of their hierarchical importance.
            :param cases: Boolean flag indicating if the vectors represent cases (default True).
            :param retain: Boolean flag to determine whether to include additional scoring in the similarity (default False).

            :return: A float representing the computed similarity between the two vectors.
        """
        #v1 is the reference (case) one. The one which we compute the similarity from
        diss = 0
        w = 0
        c2 = v2
        if cases:
            v1 = v1.get_problem_vector()
            v2 = v2.get_problem_vector()

        for i in range(len(v1)):
            #Weight by hierarchical importance (statistically calculated)
            if cases:
                w_t = ((len(features_ordered)-(i+1)+1)/len(features_ordered))
            else:
                if i <= 9:
                    #For genres to have the same weight
                    w_t = 1
                else:
                    #Otherwise, as if it was level 2 (within the tree)
                    w_t = ((len(features_ordered)-(i-8)+1)/len(features_ordered))
            
            #Quantitative variables: Euclidean 1-Dimensional distance
            if str(v1[i]).isnumeric() and (v1[i] != 0) and (v1[i] != 1):
                if cases:
                    if i == 5:
                        #Year of birth variable
                        diss += w_t*(abs(v1[i]-v2[i])/self.range_year) #Canviar i treure global
                    else:
                        diss += w_t*(abs(v1[i]-v2[i])/self.range_pages)
                else:
                    #Book's pages
                    if i == 11:
                        diss += w_t*(abs(v1[i]-v2[i])/self.range_pages_books)
                    #Publication year
                    elif i == 15:
                        diss += w_t*(abs(v1[i]-v2[i])/self.any_publi_range)
                    #Minimum age
                    elif i == 16:
                        diss += w_t*(abs(v1[i]-v2[i])/self.edat_minima_range)


            #Categorical variables: Hamming distance
            else:
                diss += w_t*(1*(v1[i] != v2[i]))
            w += w_t
        
        if cases and not retain:
            sim = (1 - (diss/w))*0.65 + ((c2.puntuacio)/5)*0.35  #Ponderem també per l'score, que conté informació sobre la precisió d'anteriors recomanacions.
        
        elif cases and retain: 
            sim = (1 - (diss/w)) #In retain mode, no punctuation is used
        else:
            sim = (1 - (diss/w)) #Podriem ponderar-ho per la valoració mitjana del llibre també

        return sim
    
    def calculate_new_score(self, subset_similar_cases):
        """
        Calculates a new score for the best case in a subset of similar cases based on their scores.
            :param subset_similar_cases: A list of cases, each with an associated score (puntuacio).
            
            :return: The case with the maximum score after recalculating its score using a specific formula.
        """
        list_similar_sorted = sorted(subset_similar_cases, key=lambda x:x.puntuacio,reverse=True)
        
        case_max_score = list_similar_sorted[0]
        max_score = case_max_score.puntuacio
        list_scores_remove = [case.puntuacio for case in list_similar_sorted[1:]]
        
        # Calculate the new score: FUNCTION = 0.8*best_case + sum((0.2/num_worst_cases)*all_worst_cases)
        new_score = 0.8*max_score + 0.2/len(list_scores_remove)*np.sum(list_scores_remove)
        case_max_score.set_puntuacio(new_score) #Li posem el nou score
        
        return case_max_score
    
    def get_trace_data(self):
        """
        Returns a list of dictionaries, each containing trace information for a specific case.
            Each dictionary contains the following keys:
            - Case_description: A dictionary containing the main characteristics of the retrieved case, with modifications to
                                display the "any_naixement" (birth year) instead of "< 2003" or ">= 2003".
            - Similarity_case: A float indicating the exact similarity value (calculated using a custom similarity function)
                               between the retrieved case and the problem case (the current user's case).
            - Old_solution: The id_llibre (book ID) of the solution given to the retrieved case.
            - Old_solution_name: The title of the solution given to the retrieved case.
            - Old_solution_description: A dictionary containing all attributes from the "Books.csv" database for the solution given to
                                       the retrieved case.
            - Broken_restrictions: A list of strings representing the restrictions that the solution given to the retrieved case
                                   does not meet. Possible values are ["Preferred Genres", "Minimum Age", "Book Language", "Not Previously Purchased"].
                                   If empty, it means the solution given to the retrieved case is the final solution (no adaptation is needed).
            - New_solution: The id_llibre (book ID) of the adapted solution, if adaptation is performed. None otherwise.
            - New_solution_name: The title of the adapted solution, if adaptation is performed. None otherwise.
            - New_solution_description: A dictionary containing all attributes from the "Books.csv" database for the adapted solution.
                                       None if no adaptation is performed.
            - Similarity_book: A float indicating the exact similarity value (calculated using the custom similarity function) between
                               the book of the solution given to the retrieved case and the book representing the adapted solution.

            :return: A list of dictionaries, each containing trace information for a specific case.
        """
        
        for c in self.reused_cases_sim:

            #Afegim també el nom del llibre1

            c['Old_solution_name'] = self.books['titol'].iloc[c['Old_solution']]
            c['Old_solution_description'] = self.books.iloc[c['Old_solution']].to_dict()
            
            #Facilitar mostra de Gèneres
            generes = []
            c_dict = c["Case_description"]
            for g in self.genre_options:
                #Si té aquell gènere com a preferència (1)
                if c_dict[g] == 1:
                    generes.append(g)
            c["Case_description"]['generes_list'] = generes

            #Broken Restrictions Translating and New Book Description only for adapted solutions
            
            if len(c['Broken_restrictions']) > 0:
                broken = ["Gèneres preferits", "Edat mínima", "Idioma del llibre", "No recomanat anteriorment"]
                c['Broken_restrictions'] = [broken[i] for i in range(len(c['Broken_restrictions'])) if not c['Broken_restrictions'][i]]
                c['New_solution_name'] = self.books['titol'].iloc[c['New_solution']]
                c['New_solution_description'] = self.books.iloc[c['New_solution']].to_dict()
            else:
                c['New_solution_name'] = None
                c['New_solution_description'] = None

        return self.reused_cases_sim

    ##########################################
    ############## THE FOUR R's ##############
    ##########################################


    def retrieve(self,data):
        """
        Retrieves cases from the case base based on user-provided data.
            :param data: A dictionary containing user-selected attributes.

            :return: A set of cases that meet the filtering criteria specified by the user in the data dictionary.
                     If this set has fewer than three cases, it searches for the nearest leaf nodes in the tree and adds their cases until there are at least three cases.
        """
        subset_cases_node, list_parents = self.case_base.evaluate_case_through_tree(data,[self.case_base])

        #Si el node fulla corresponent no té prou casos (mínim 3) cerquem
        #als nodes fulla més propers i n'afegim els casos necessaris.
        subset_cases = subset_cases_node._subset.copy()
        co = 1

        ll_casos_idx = []

        while len(subset_cases) < 3:
            nearest_leaf_nodes = list_parents[-co].get_leaves()
            for n in nearest_leaf_nodes:
                for n_case in n._subset: 
                    subset_cases.add(n_case)
                    ll_casos_idx.append(n_case.id_cas)
            co += 1

        self.retrieved = subset_cases


    def reuse(self, data):
        """
        Reuses the solutions given to the 3 most similar cases from the retrieved cases.
            :param data: A dictionary containing user-selected attributes.

            :return: Three solutions that were given to the 3 most similar cases.
        """
        self.case_problem = Case()
        self.case_problem.from_dict(data)
        sim_list = []
        for case_2 in self.retrieved:
            sim_list.append((case_2,self.similarity(self.case_problem,case_2,self.features_by_importance),case_2.id_cas))
        
        sim_list_sorted = sorted(sim_list, key=lambda x:x[1],reverse=True)

        #To trace
        self.reused_cases_sim = []
        for result in sim_list_sorted[:3]:
            case_dct = result[0].to_dict()
            case_dct['any_naixement'] = result[0].any_naixement #Per a què sigui numèric
            self.reused_cases_sim.append({'Case_description':case_dct,'Similarity_case':result[1],'Old_solution':case_dct['solucio']})
        self.reused = (sim_list_sorted[0][0].solucio,sim_list_sorted[1][0].solucio,sim_list_sorted[2][0].solucio)



    def revise(self,book,index):
        """
        Adapt (if necessary) each of the three solutions given so that they meet all the mandatory restrictions of the new problem, as well as being as similar as possible to the solution retrieved for the most similar case found.
            :param book: The book to be revised and adapted.
            :param index: The index of the case being revised.

            :return: The adapted book ID.
        """
        #Comprovació de restriccions obligatòries
        oblig = [False,False,False,False]
        book_row = self.books[self.books['id_llibre'] == book]

        #Gènere del Llibre
        
        v = self.case_problem.get_problem_vector()
        genres_ordered_vector = self.genre_options[0:2]+[None]+self.genre_options[2:]
        genres_case = set()
        for i in range(len(genres_ordered_vector)):
            g = genres_ordered_vector[i]
            if v[3+i] == 1:
                genres_case.add(g)
        intersection = book_row.iloc[0]['pertany_a'].intersection(genres_case)
        

        if len(intersection) > 0:
            oblig[0] = True
        
        #Edat mínima

        if book_row.iloc[0]['edat_minima'] <= (2023-v[5]):
            oblig[1] = True

        #Idioma

        if len(self.case_problem.idioma.intersection(book_row.iloc[0]['traduccions'])) > 0:
            oblig[2] = True

        #Llibre recomanat anteriorment

        if len(self.cases[(self.cases['id_usuari'] == self.case_problem.id_usuari) & (self.cases['id_llibre'] == book)]) == 0:
            oblig[3] = True

        #Si és tot True, hem acabat l'adaptació (no fa falta). Si no, agafar els llibres que compleixin les obligatòries i que siguin més 
        #similars a les preferències de l'usuari (?)

        if False in oblig:

            #For trace
            self.reused_cases_sim[index]['Broken_restrictions'] = oblig

            #Subset de llibres que compleixen les restriccions obligatòries

            #Edat mínima
            subset_books_edat = self.books[self.books['edat_minima'] <= (2023-v[5])]

            #Idioma
            mask = np.array([len(set_idioma.intersection(self.case_problem.idioma)) > 0 for set_idioma in subset_books_edat['traduccions'].values])
            subset_books_idioma = subset_books_edat[mask]

            #No recomanat anteriorment

            mask = np.array([len(self.cases[(self.cases['id_usuari'] == self.case_problem.id_usuari) & (self.cases['id_llibre'] == id_llibre)]) == 0 for id_llibre in subset_books_idioma['id_llibre'].values])
            subset_books_no_llegit = subset_books_idioma[mask]

            #Gènere del llibre
            mask = np.array([len(set_genere.intersection(genres_case)) > 0 for set_genere in subset_books_no_llegit['pertany_a'].values]) 
            subset_books_genere = subset_books_no_llegit[mask]

            #Si l'últim filtre resulta en un DF buit, aleshores mirem el filtre d'adalt, que assumim no serà buit (és molt poc restrictiu).

            if len(subset_books_genere) == 0:
                subset_books = subset_books_no_llegit.copy()
            else:
                subset_books = subset_books_genere.copy()

            #Ara busquem, del subset, el llibre més similar a la solució retrieved basant-nos en la mesura de similitud que hem creat
            #amb els atributs [Genere, any_publicacio, best_seller, saga, adaptacio_a_pelicula, edat_minima, num_pagines] en aquest ordre d'importància (quedaran ponderats per l'ordre)
            #i també per la valoració general del llibre.

            #Si només queda un llibre, el retornem directament. En altre cas, retornem el que doni una mesura de similitud major
            books_sim = []

            if len(subset_books) == 1:
                return int(subset_books['id_llibre'])
            else:
                
                
                vector_llibre_adaptat = [1 if genere in book_row.iloc[0]['pertany_a'] else 0 for genere in self.genre_options]
                vector_llibre_adaptat += [book_row.iloc[0]['any_publicacio'],book_row.iloc[0]['best_seller'],book_row.iloc[0]['saga'],book_row.iloc[0]['adaptacio_a_pelicula'],book_row.iloc[0]['edat_minima'],book_row.iloc[0]['num_pagines']]
                for idx,fila in subset_books.iterrows():
                    vector_llibre_2 = [1 if genere in fila['pertany_a'] else 0 for genere in self.genre_options]
                    vector_llibre_2 += [fila['any_publicacio'],fila['best_seller'],fila['saga'],fila['adaptacio_a_pelicula'],fila['edat_minima'],fila['num_pagines']]
                    books_sim.append((fila['id_llibre'],self.similarity(vector_llibre_adaptat,vector_llibre_2,cases=False,features_ordered=self.genre_options+['any_publicacio', 'best_seller', 'saga', 'adaptacio_a_pelicula', 'edat_minima', 'num_pagines'])))

                books_sim_sorted = sorted(books_sim, key=lambda x:x[1],reverse=True)

                #Avoid returning an already adapted book (ready to be recommended)
                adapted = books_sim_sorted[0][0]
                a = 1
                while (adapted in self.adapted_solutions):
                    adapted = books_sim_sorted[a][0]
                    a += 1
                
                #For trace
                self.reused_cases_sim[index]['New_solution'] = adapted
                self.reused_cases_sim[index]['Similarity_book'] = books_sim_sorted[a-1][1]
                return adapted
        else:
            #For trace
            self.reused_cases_sim[index]['Broken_restrictions'] = []
            self.reused_cases_sim[index]['New_solution'] = None
            self.reused_cases_sim[index]['Similarity_book'] = None
            return book
        
        
    def retain(self, case):
        """
        Retains the new case by adding it to the case base. This function is used to store the experience from
        the current interaction, including the user's preferences, the recommended book, and the user's response
        (purchase and rating), for future use in the case-based reasoning process.
            :param id_user: User's identifier.
            :param data: The dataset containing user preferences. This is typically a dictionary where each key-value
                         pair corresponds to a user preference or response.
            :param id_book: The identifier of the recommended book.
            :param rate: The user's rating for the recommended book.
            :param comprat: A binary variable indicating whether the book was purchased (Yes/No).
        """
        storing_case = case #En principi, afegim el cas adaptat

        #Modify csv
        valors_genere = [case.ciencia_ficcio, case.historica, case.ficcio, case.comedia, case.romance, case.fantasia, case.ciencia, case.creixement_personal, case.policiaca, case.juvenil]
        self.cases.loc[len(self.cases)] = [case.id_usuari, case.solucio, case.puntuacio, case.genere_persona, case.any_naixement, case.pref_adaptacio_peli, case.pref_best_seller,
                                           case.pref_tipus_lectura, case.pref_sagues,case.comprat] + valors_genere + [case.pagines_max, case.idioma]
        

        valoracio = self.books[self.books['id_llibre'] == case.solucio]['valoracio']
        num_lectors = self.books[self.books['id_llibre'] == case.solucio]['num_lectures']
        
        
        # Calcul nova valoracio mitjana = val actual de llibre recomanat*
        # Sumar +1 perque si el lector nou l'ha comprat hi ha un lector mes que ho ha llegit


        if case.comprat == "Si":
            nova_valoracio = (valoracio*num_lectors + case.puntuacio) / (num_lectors+1)
            self.books.at[case.solucio, 'num_lectures'] = num_lectors + 1
            self.books.at[case.solucio, 'valoracio'] = nova_valoracio
        
        # Data = problem description /// case_base = decision tree
        leaf_node, _ = self.case_base.evaluate_case_through_tree(case.to_dict(), [self.case_base])
        
        # Look for the ones that have the same book recommended, id_book, inside the leaf

        subset_cases = leaf_node._subset.copy() 
        if len(subset_cases) >= 20:
            sim_id_book_cases = [] # subset_cases
            for case2 in subset_cases:
                if case2.solucio == case.solucio:
                    sim_id_book_cases.append(case2)
            
            # Compare the new case to each one of subset_cases
     
            list_similar_cases = []
            for case2 in sim_id_book_cases:
                sim = self.similarity(self.case_problem, case2, retain=True, features_ordered=self.features_by_importance)
                if  sim >= 0.90:
                    list_similar_cases.append(case2)
                    subset_cases.remove(case2)
            #i afegim el nou cas també, ja que forma part del "cluster" de casos semblants
            list_similar_cases.append(case)
            
            # Return the instant_best_score_case, calculated_score, redundant_cases
            # IMP: The storing_case has been kicked out of the redundant_cases
            #      storing_case: Case to store  //  redundant_cases: Cases to remove

            if len(list_similar_cases) > 1:
                #Si no hi ha cap similar, no fem el càlcul de retenció
                storing_case = self.calculate_new_score(list_similar_cases)
            
            # We need to assure that the 'storing_case' is not in the self.cases and update or add
            subset_cases.add(storing_case) # storing_case: instance of Case() class
        else:
            subset_cases.add(storing_case)

        self.case_base.modify_leaf(leaf_node._path, subset_cases)
        #self.case_base.plot_tree()

        # Un cop funcioni el recomanador, amb la seguent linía guardarem els casos al csv permanentment i books
        #COMENTAR SI NO ES VOL GUARDAR.

        self.cases.to_csv('../ontologia/Cases.csv', encoding='utf-8', index=False)
        self.books.to_csv('../ontologia/Books.csv', encoding='utf-8', index=False)
        self.users.to_csv('../ontologia/Users.csv', encoding='utf-8', index=False)

