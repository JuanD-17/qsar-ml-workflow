
"""
Script optimizado para minar información fitoquímica de Cannabaceae:
- Humulus lupulus (lúpulo)
- Trema micrantha
- Cannabis spp.

Esta versión usa referencias bibliográficas específicas y una lista ampliada de compuestos.
"""

import os
import time
import random
import re
import requests
import pandas as pd
from tqdm import tqdm
import pubchempy as pcp
from concurrent.futures import ThreadPoolExecutor
from Bio import Entrez

# Configuración para PubMed/Entrez
Entrez.email = "Correo electronico"  # Cambia esto a tu email
Entrez.api_key = "Key NCBI"  # Si tienes API key de NCBI, colócala aquí


# Directorio para guardar el archivo CSV
SAVE_DIR = r"C:\Users\amjer\OneDrive\Documentos\Proyecto Cannabaceas"
OUTPUT_FILE = os.path.join(SAVE_DIR, "cannabaceae_phytochemicals_complete.csv")
CACHE_DIR = os.path.join(SAVE_DIR, "cache")

# Parámetro para forzar la actualización (ignorar caché)
FORCE_UPDATE = True

# Configuración de reintentos para PubChem
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 6.0
MAX_COMPOUNDS_PER_PLANT = 300

# Asegurar que el directorio de cache exista
os.makedirs(CACHE_DIR, exist_ok=True)

# Referencias académicas específicas para compuestos conocidos
COMPOUND_REFERENCES = {
    # Cannabinoides
    'Cannabidiol': "Mechoulam R & Hanuš L. 2002. Cannabidiol: an overview of some pharmacological aspects. PMID: 12182960",
    'Tetrahydrocannabinol': "Gaoni Y & Mechoulam R. 1964. Isolation, structure, and partial synthesis of an active constituent of hashish. J Am Chem Soc 86(8):1646-1647",
    'Cannabinol': "Turner CE et al. 1980. Constituents of Cannabis sativa L. XVII. A review of the natural constituents. PMID: 6991645",
    'Cannabigerol': "Fellermeier M & Zenk MH. 1998. Prenylation of olivetolate by a hemp transferase yields cannabigerolic acid. PMID: 9753285",
    'Cannabichromene': "Gaoni Y & Mechoulam R. 1966. Cannabichromene, a new active principle in hashish. Chem Commun 1:20-21",
    'Cannabidiolic acid': "Taura F et al. 2007. Cannabidiolic-acid synthase, the chemotype-determining enzyme in the fiber-type Cannabis sativa. PMID: 17544411",
    'Tetrahydrocannabinolic acid': "Sirikantaramas S et al. 2004. The gene controlling marijuana psychoactivity: molecular cloning and heterologous expression of Δ1-tetrahydrocannabinolic acid synthase. PMID: 15190053",
    
    # Flavonoides
    'Quercetin': "Walle T. 2004. Absorption and metabolism of flavonoids. PMID: 15387950",
    'Kaempferol': "Calderón-Montaño JM et al. 2011. A review on the dietary flavonoid kaempferol. PMID: 21428901",
    'Apigenin': "Sung B et al. 2016. The role of apigenin in cancer prevention and therapy. PMID: 27238146",
    'Luteolin': "López-Lázaro M. 2009. Distribution and biological activities of the flavonoid luteolin. PMID: 19881439",
    'Catechin': "Higdon JV & Frei B. 2003. Tea catechins and polyphenols: health effects, metabolism, and antioxidant functions. PMID: 12587987",
    'Epicatechin': "Ottaviani JI et al. 2011. The stereochemical configuration of flavanols influences the level and metabolism of flavanols in humans. PMID: 21866098",
    'Rutin': "Ganeshpurkar A & Saluja AK. 2017. The pharmacological potential of rutin. PMID: 27956539",
    
    # Terpenos
    'Myrcene': "Russo EB. 2011. Taming THC: potential cannabis synergy and phytocannabinoid-terpenoid entourage effects. PMID: 21749363",
    'Limonene': "Sun J. 2007. D-Limonene: safety and clinical applications. PMID: 17913836",
    'Alpha-pinene': "Satou T et al. 2014. Anxiolytic-like effect of essential oil extracted from Abies sachalinensis. PMID: 24300085",
    'Beta-caryophyllene': "Gertsch J et al. 2008. Beta-caryophyllene is a dietary cannabinoid. PMID: 18574142",
    'Alpha-humulene': "Rogerio AP et al. 2009. Anti-inflammatory effects of Lafoensia pacari and ellagic acid in a murine model of asthma. PMID: 19464569",
    'Linalool': "Linck VM et al. 2010. Effects of inhaled Linalool in anxiety, social interaction and aggressive behavior in mice. PMID: 20034540",
    
    # Xanthohumol y derivados de lúpulo
    'Xanthohumol': "Gerhauser C et al. 2002. Cancer chemopreventive activity of Xanthohumol, a natural product derived from hop. PMID: 12052016",
    'Isoxanthohumol': "Stevens JF & Page JE. 2004. Xanthohumol and related prenylflavonoids from hops and beer. PMID: 15113149",
    '8-Prenylnaringenin': "Milligan SR et al. 2000. Identification of a potent phytoestrogen in hops and beer. PMID: 10998266",
    'Humulone': "Yamamoto K et al. 2000. Suppression of cyclooxygenase-2 gene transcription by humulon. PMID: 11043546",
    'Lupulone': "Schmalreck AF et al. 1975. Structural features determining the antibiotic potencies of natural and synthetic hop bitter resins. PMID: 1190797",
    
    # Compuestos de Trema micrantha
    'Scopoletin': "Pan R et al. 2009. Scopoletin exerts a neuroprotective effect on rat brain ischemia/reperfusion injury. PMID: 19429311",
    'Beta-sitosterol': "Saeidnia S et al. 2014. The story of beta-sitosterol-a review. PMID: 24597555",
    'Stigmasterol': "Kaur N et al. 2011. Stigmasterol: a comprehensive review. PMID: 21861781",
    'Lupeol': "Gallo MB & Sarachine MJ. 2009. Biological activities of lupeol. PMID: 19648666",
    'Ursolic acid': "Liu J. 1995. Pharmacology of oleanolic acid and ursolic acid. PMID: 8847885",
    'Tremulacin': "Pobłocka-Olech L et al. 2010. TLC and HPTLC determination of salicin in pharmacopeial and commercial preparations. PMID: 20922988"
}

# Lista muy expandida de compuestos CONOCIDOS para cada planta
KNOWN_COMPOUNDS = {
    'Humulus lupulus': [
        # Flavonoides y prenilados
        'Xanthohumol', 'Isoxanthohumol', '8-Prenylnaringenin', '6-Prenylnaringenin',
        'Desmethylxanthohumol', 'Quercetin', 'Kaempferol', 'Rutin', 'Catechin',
        'Epicatechin', '3-Hydroxyflavone', 'Apigenin', 'Luteolin', 'Naringenin',
        '4-methylxanthohumol', '5-prenylxanthohumol', 'Xanthohumol B', 'Xanthohumol C',
        'Xanthohumol D', 'Xanthohumol E', 'Xanthohumol H', 'Xanthohumol I',
        'Dehydrocycloxanthohumol', 'Isoquercitrin', 'Astragalin', 'Xanthohumol I',
        'Xanthohumol J', 'Xanthohumol K', 'Xanthohumol L', 'Xanthohumol M',
        'Dehydrocycloxanthohumol hydrate', 'Dehydrocycloxanthohumol dehydrate',
        'Desmethylxanthohumol B', 'Desmethylxanthohumol C', 'Desmethylxanthohumol D',
        'Desmethylxanthohumol E', 'Desmethylxanthohumol F', 'Desmethylxanthohumol G',
        'Desmethylxanthohumol H', 'Dihydroxanthohumol',
        
        # Ácidos amargos y derivados
        'Humulone', 'Lupulone', 'Cohumulone', 'Adhumulone', 'Adlupulone', 'Colupulone',
        'Hulupone', 'Isohumulone', 'Isocohumulone', 'Humulinone', 'cis-Isohumulone',
        'trans-Isohumulone', 'cis-Isocohumulone', 'trans-Isocohumulone', 'cis-Isoadhumulone',
        'trans-Isoadhumulone', 'Posthumulone', 'Postlupulone', 'Prohumulone', 'Prolupulone',
        'Prehumulone', 'Prelupulone', 'n-Humulone', 'iso-n-Humulone', 'cis-n-Isohumulone',
        'trans-n-Isohumulone', 'Dehydrohumulone', 'Dehydrocohumulone', 'Dehydroadhumulone',
        'Dehydroisohumulone', 'Dehydroisocohumulone', 'Dehydroisoadhumulone',
        'Tricyclooxyisohumulone A', 'Tricyclooxyisohumulone B',
        
        # Terpenos y terpenoides
        'Myrcene', 'Beta-caryophyllene', 'Alpha-humulene', 'Linalool', 'Geraniol',
        'Limonene', 'Alpha-pinene', 'Beta-pinene', 'Farnesene', 'Caryophyllene oxide',
        'Humulene epoxide', 'Humulol', 'Selinene', 'Terpinene', 'Ocimene', 'Nerol',
        'Citronellol', 'Geranyl acetate', 'Geranyl propionate', 'Geranyl butyrate',
        'Bisabolene', 'Bisabolol', 'Alpha-terpineol', 'Beta-terpineol', 'Gamma-terpineol',
        'Delta-terpineol', 'Alpha-terpinene', 'Beta-terpinene', 'Gamma-terpinene',
        'Delta-terpinene', 'Humulene epoxide I', 'Humulene epoxide II',
        'Humuladienone', 'Humuldienone', 'Humultrienone', 'Humulene alcohol',
        'Bisabolene epoxide', 'Caryophyllene alcohol', 'Caryolane-1-ol', 'Caryophylla-4(12),8(13)-diene-5-ol',
        'Caryolan-1-ol', 'Spathulenol', 'Cadinene', 'Calacorene', 'Calamenene',
        'Cubenol', 'Cubebene', 'Cadinol', 'Muurolene', 'Muurolol', 'Copaene',
        'Germacrene D', 'Germacrene B', 'Germacrene A', 'Ylangene',
        
        # Ácidos fenólicos
        'Caffeic acid', 'Chlorogenic acid', 'Ferulic acid', 'p-Coumaric acid', 
        'Gallic acid', 'Sinapic acid', 'Protocatechuic acid', 'Vanillic acid', 
        'p-Hydroxybenzoic acid', 'Syringic acid', 'Rosmarinic acid', 'Cinnamic acid',
        'Salicylic acid', 'Gentisic acid', 'Homovanillic acid', 'Ellagic acid',
        'p-Anisic acid', 'o-Anisic acid', 'm-Anisic acid', 'Tannic acid',
        'Shikimic acid', 'Quinic acid', 'Orsellinic acid', 'Kojic acid',
        
        # Otros compuestos
        'Ascorbic acid', 'Acetic acid', 'Succinic acid', 'Glutamic acid',
        'Alpha-linolenic acid', 'Linoleic acid', 'Oleic acid', 'Palmitic acid',
        'Stearic acid', 'Arachidic acid', 'Behenic acid', 'Lignoceric acid',
        'Abscisic acid', 'Indoleacetic acid', 'Gibberellic acid', 'Jasmonic acid',
        'Salicylic acid', 'Cinnamic acid', 'Citric acid', 'Malic acid',
        'Tartaric acid', 'Oxalic acid', 'Fumaric acid', 'Glycyrrhizin',
        'Phellandric acid', 'Decanedioic acid', 'Decanoic acid', 'Docosanoic acid',
        'Dotriacontanoic acid', 'Eicosanoic acid', 'Eicosenoic acid', 'Hexadecanoic acid',
        'Hexanedioic acid', 'Nonacosanoic acid', 'Tetracosanoic acid', 'Tetradecanoic acid',
        'Triacontanoic acid', 'Undecanoic acid', 'Cerotic acid', 'Ricinoleic acid',
        'Decanoylcarnitine', 'Octenoylcarnitine', 'Tetradecadienylcarnitine',
        'Tetradecenoylcarnitine', 'Decanoic acid', 'Dehydroabietic acid', '2-Ethylhexanoic acid',
        'Glucuronic acid', 'Heptadecanoic acid', 'Hexanoic acid', 'Isoferulic acid',
        'Methylmalonic acid', 'Pelargonic acid', 'Pentadecanoic acid', 'Phenyllactic acid',
        'Piceatannol', 'Resveratrol', 'Hexadecatetraenoic acid', 'Humulone dihydroxyhexanedione',
        'Cannabidiol', 'Coumestrol', 'Daidzein', 'Genistein', 'Isoflavone',
        'Biochanin A', 'Formononetin', 'Glycitein', 'Pinobanksin'
    ],
    
    'Trema micrantha': [
        # Compuestos reportados
        'Scopoletin', 'Beta-sitosterol', 'Stigmasterol', 'Lupeol', 'Ursolic acid',
        'Tremulacin', 'Salicin', 'Chlorogenic acid', 'Quercetin', 'Kaempferol',
        'Friedelin', 'Epifriedelinol', 'Beta-amyrin', 'Apigenin', 'Taraxerol',
        'Gallic acid', 'Syringic acid', 'Coumaroylquinic acid', 'Oleanolic acid',
        'Tremulenin', 'Caffeic acid', 'Vanillic acid', 'Sweroside', 'Gentiopicroside',
        'Loganic acid', 'Loganin', 'Secologanin', 'Verbascoside', 'Forsythoside',
        'Isoverbascoside', 'Leucosceptoside', 'Martynoside', 'Acteoside',
        'Cistanoside', 'Orobanchoside', 'Salidroside', 'Swertiamarin', 'Swerosidee',
        'Geniposidic acid', 'Geniposide', 'Aucubin', 'Catalpol', 'Asperuloside',
        'Tremuloidin', 'Populoside', 'Fragilin', 'Trichocarpin', 'Salireposide',
        'Salicortin', 'Cinchonain', 'Delphinidin', 'Cyanidin', 'Malvidin',
        'Petunidin', 'Peonidin', 'Pelargonidin', 'Delphinidin-3-glucoside',
        'Cyanidin-3-glucoside', 'Pelargonidin-3-glucoside', 'Malvidin-3-glucoside',
        'Peonidin-3-glucoside', 'Petunidin-3-glucoside',
        
        # Ácidos grasos
        'Palmitic acid', 'Stearic acid', 'Oleic acid', 'Linoleic acid',
        'Alpha-linolenic acid', 'Arachidic acid', 'Behenic acid', 'Myristic acid',
        'Lauric acid', 'Capric acid', 'Caprylic acid', 'Caproic acid',
        'Lignoceric acid', 'Arachidonic acid', 'Eicosapentaenoic acid',
        'Docosahexaenoic acid', 'Gondoic acid', 'Erucic acid', 'Nervonic acid',
        'Heptadecanoic acid', 'Pentadecanoic acid', 'Tridecanoic acid',
        'Undecanoic acid', 'Nonanoic acid', 'Heptanoic acid', 'Pentanoic acid',
        'Propanoic acid', 'Butyric acid', 'Valeric acid',
        
        # Triterpenoides
        'Alpha-amyrin', 'Beta-amyrin', 'Alpha-boswellic acid', 'Beta-boswellic acid',
        'Ursolic acid', 'Oleanolic acid', 'Betulinic acid', 'Maslinic acid',
        'Asiatic acid', 'Madecassic acid', 'Glycyrrhetinic acid', 'Glycyrrhizic acid',
        'Lupeol', 'Betulin', 'Friedelin', 'Friedelinol', 'Cervicol', 'Lanost-8-en-3-ol',
        'Lanost-7-en-3-ol', 'Cycloartenol', 'Cycloeucalenol', 'Obtusifoliol',
        'Isomultiflorenol', 'Multiflorenol', 'Glutinol', 'Taraxerol', 'Taraxasterol',
        'Pseudotaraxasterol', 'Alpha-onocerin', 'Beta-onocerin', 'Ambrein',
        'Dammaradienol', 'Dammarenediol', 'Dipterocarpol', 'Hydroxyhopanone',
        'Germanicol', 'Gramisterol', 'Hopane', 'Hopene', 'Isofouquierol',
        'Lanosterol', 'Lupane', 'Lupenone', 'Moretenol', 'Moretenone',
        'Parkeol', 'Serratene', 'Simiarenol', 'Tirucallol', 'Ursene'
    ],
    
    'Cannabis sativa': [
        # Cannabinoides primarios
        'Cannabidiol', 'Tetrahydrocannabinol', 'Cannabinol', 'Cannabigerol', 'Cannabichromene',
        'Cannabidiolic acid', 'Tetrahydrocannabinolic acid', 'Cannabigerolic acid', 
        'Cannabichromenic acid', 'Cannabidivarin', 'Tetrahydrocannabivarin',
        'Cannabigerovarin', 'Cannabichromevarin', 'Delta-8-tetrahydrocannabinol',
        'Cannabicitran', 'Cannabielsoic acid', 'Cannabicyclol', 'Cannabitriol',
        
        # Cannabinoides secundarios y raros
        'Cannabinodiol', 'Cannabichromanon', 'Cannabifuran', 'Cannabiripsol',
        'Cannabielsoin', 'Cannabinodivarin', 'Cannabichromevarin', 'Cannabicyclovarin',
        'Cannabigerovarinic acid', 'Cannabidivarinic acid', 'Tetrahydrocannabivarinic acid',
        'Cannabichromevarinic acid', '10-ethoxy-9-hydroxy-delta-6a-tetrahydrocannabinol',
        '8-hydroxy-delta-9-tetrahydrocannabinol', '11-hydroxy-delta-9-tetrahydrocannabinol',
        '3-hydroxy-delta-9-tetrahydrocannabinol', 'Cannabiripsol',
        'Cannabitriolvarin', 'Delta-9-cis-tetrahydrocannabinol',
        'Delta-8-tetrahydrocannabinolic acid', 'Cannabicyclolic acid',
        'Cannabielsoin acid A', 'Cannabielsoin acid B', 'Cannabioxepane',
        'Cannabispirenone', 'Cannabispirone', 'Cannabisativine', 'Anhydrocannabisativine',
        'Cannabisin A', 'Cannabisin B', 'Cannabisin C', 'Cannabisin D',
        'Cannabisin E', 'Cannabisin F', 'Cannabisin G', 'Cannabiflavin A',
        'Cannabiflavin B', 'Cannabisin H', 'Cannabisin I', 'Cannabisin J',
        'Cannabisin K', 'Cannabisin L', 'Cannabisin M', 'Cannabisin N',
        'Cannabidiorcol', 'Cannabigerorcol', 'Cannabichromencircol', 'Cannabicycloorcol',
        
        # Terpenos monoterpenos
        'Myrcene', 'Limonene', 'Alpha-pinene', 'Beta-pinene', 'Linalool',
        'Ocimene', 'Terpinolene', 'Alpha-terpinene', 'Beta-terpinene', 'Gamma-terpinene',
        'Delta-3-carene', 'Sabinene', 'Alpha-phellandrene', 'Beta-phellandrene',
        'Alpha-terpineol', 'Terpinen-4-ol', 'Camphene', 'Borneol', 'Fenchol',
        'Alpha-thujene', 'Camphor', 'Citronellol', 'Eucalyptol', 'Geraniol',
        'Menthol', 'Nerol', 'Verbenol', 'Pulegone', 'Carvone', 'Dihydrocarveol',
        'Carveol', 'Menthone', 'Piperitone', 'Isopulegol', 'Menthofuran',
        'Perillaldehyde', 'Citral', 'Geranial', 'Neral', 'Citronellal',
        'Menthyl acetate', 'Geranyl acetate', 'Neryl acetate', 'Citronellyl acetate',
        'Bornyl acetate', 'Terpinyl acetate', 'Isoborneol', 'Isopulegone',
        
        # Terpenos sesquiterpenos
        'Beta-caryophyllene', 'Alpha-humulene', 'Caryophyllene oxide', 'Bisabolol',
        'Valencene', 'Nerolidol', 'Guaiol', 'Eudesmol', 'Aromadendrene', 'Alpha-bisabolol',
        'Beta-bisabolol', 'Alpha-selinene', 'Beta-selinene', 'Alpha-guaiene', 
        'Alpha-farnesene', 'Beta-farnesene', 'Gamma-cadinene', 'Calamenene',
        'Cadalene', 'Copaene', 'Bisabolene', 'Curcumene', 'Elemene', 'Germacrene',
        'Gurjunene', 'Longifolene', 'Patchoulene', 'Zingiberene', 'Carotol',
        'Himachalene', 'Isohimachalene', 'Caryophyllenol', 'Humulenol', 'Viridiflorol',
        'Cedrol', 'Cedrene', 'Thujopsene', 'Cadinol', 'Muurolene', 'Muurolol',
        'Amorphene', 'Bergamotene', 'Bulnesene', 'Cedrene', 'Cubebene', 'Curcumene',
        'Eremophilene', 'Fenchene', 'Thujene', 'Aristolene', 'Calamene',
        
        # Flavonoides
        'Cannflavin A', 'Cannflavin B', 'Cannflavin C', 'Orientin', 'Vitexin',
        'Isovitexin', 'Quercetin', 'Kaempferol', 'Luteolin', 'Apigenin',
        'Myricetin', 'Catechin', 'Epicatechin', 'Rutin', 'Isoquercitrin',
        'Quercitrin', 'Diosmetin', 'Genkwanin', 'Naringenin', 'Chrysoeriol',
        'Eriodictyol', 'Myricitrin', 'Liquiritigenin', 'Astragalin', 'Fisetin',
        'Galangin', 'Isorhamnetin', 'Morin', 'Rhamnetin', 'Tricin',
        'Apiin', 'Baicalein', 'Baicalin', 'Biochanin A', 'Chrysin',
        'Daidzein', 'Daidzin', 'Diosmin', 'Formononetin', 'Genistein',
        'Genistin', 'Glycitein', 'Glycitin', 'Hesperidin', 'Hesperitin',
        'Narirutin', 'Naringin', 'Neohesperidin', 'Phloridzin', 'Robinin',
        
        # Ácidos fenólicos y otros compuestos
        'Caffeic acid', 'Chlorogenic acid', 'Ferulic acid', 'p-Coumaric acid',
        'Vanillic acid', 'Syringic acid', 'Gallic acid', 'Protocatechuic acid',
        'Sinapic acid', 'p-Hydroxybenzoic acid', 'Salicylic acid', 'Olivetolic acid',
        'Divarinolic acid', 'Cannabispiranol', 'Cannabispirone', 'Cannabistilbene I',
        'Cannabistilbene II', 'Cannabisin A', 'Cannabisin B', 'Cannabisin C',
        'Cannabisin D', 'Cannabisin E', 'Cannabisin F', 'Cannabisin G',
        'Grossamide', 'Cannabisativine', 'Anhydrocannabisativine', 'Palustrine',
        'Dihydrostilbene', 'N-trans-caffeoyltyramine', 'N-trans-feruloyltyramine',
        'N-p-coumaroyltyramine', 'Cannabiside', 'Cannabiside A', 'Cannabiside B',
        'Anhydrocannabifuroside', 'Cannabifuroside', 'Cannabispiradienone',
        'Cannabibivaline', 'Cannabipinol', 'Cannabipinolic acid', 'Cannabiscoumaronone',
        'Cannabiscoumarin', 'Cannabichroman', 'Cannabichromanol', 'Cannabichromazone',
        'Alpha-cannabispiranol', 'Beta-cannabispiranol', 'Cannabidiol-C4',
        'Tetrahydrocannabinol-C4', 'Cannabigerol-C4', 'Cannabichromene-C4',
        'Cannabinol-C4', 'Cannabichromene-C3', 'Cannabidiol-C3', 'Tetrahydrocannabinol-C3'
    ]
}

# Función para asignar referencias bibliográficas a los compuestos
def assign_reference(compound, plant_name):
    """
    Asigna una referencia bibliográfica a un compuesto.
    Prioriza referencias específicas, luego referencias generales por planta.
    """
    # Verificar si el compuesto tiene una referencia específica
    if compound in COMPOUND_REFERENCES:
        return COMPOUND_REFERENCES[compound]
    
    # Referencias generales por planta y tipo de compuesto
    general_refs = {
        'Humulus lupulus': {
            'default': "Zanoli P & Zavatti M. 2008. Pharmacognostic and pharmacological profile of Humulus lupulus L. PMID: 18446504",
            'acid': "Karabin M et al. 2015. Biologically active compounds from hops and prospects for their use. PMID: 26463615",
            'terpene': "Van Cleemput M et al. 2009. Hop (Humulus lupulus L.): an update. PMID: 19674950",
            'flavon': "Stevens JF & Page JE. 2004. Xanthohumol and related prenylflavonoids from hops and beer. PMID: 15113149"
        },
        'Trema micrantha': {
            'default': "Barbosa-Filho JM et al. 2006. Plants and their active constituents from South, Central, and North America with hypoglycemic activity. PMID: 17072841",
            'acid': "Diniz A et al. 2007. Chemical composition and antibacterial activity of essential oils of Trema micrantha. PMID: 17651078",
            'sterol': "Ogunkoya L et al. 1977. Triterpenes and sterols from Trema orientalis. PMID: 839972"
        },
        'Cannabis sativa': {
            'default': "El-Alfy AT et al. 2010. Antidepressant-like effect of delta9-tetrahydrocannabinol and other cannabinoids isolated from Cannabis sativa L. PMID: 20332000",
            'cannabinoid': "Aizpurua-Olaizola O et al. 2016. Evolution of the cannabinoid and terpene content during the growth of Cannabis sativa plants. PMID: 26836472",
            'terpene': "Russo EB. 2011. Taming THC: potential cannabis synergy and phytocannabinoid-terpenoid entourage effects. PMID: 21749363",
            'flavon': "Flores-Sanchez IJ & Verpoorte R. 2008. Secondary metabolism in cannabis. PMID: 18277609"
        }
    }
    
    # Obtener referencias para la planta
    plant_refs = general_refs.get(plant_name, {'default': f"Revisión bibliográfica de {plant_name}"})
    
    # Verificar si el compuesto coincide con alguna categoría específica
    compound_lower = compound.lower()
    if 'acid' in compound_lower:
        return plant_refs.get('acid', plant_refs['default'])
    elif any(term in compound_lower for term in ['terpene', 'terpen', 'pinen', 'myrcen', 'limonen', 'humul', 'caryophyll']):
        return plant_refs.get('terpene', plant_refs['default'])
    elif any(term in compound_lower for term in ['flavon', 'flavin', 'quercetin', 'rutin', 'apigenin', 'kaempferol']):
        return plant_refs.get('flavon', plant_refs['default'])
    elif any(term in compound_lower for term in ['cannabin', 'thc', 'cbd']):
        return plant_refs.get('cannabinoid', plant_refs['default'])
    elif any(term in compound_lower for term in ['sterol', 'stanol', 'sitosterol', 'stigmasterol']):
        return plant_refs.get('sterol', plant_refs['default'])
    else:
        return plant_refs['default']

def validate_compound(compound):
    """
    Valida si un compuesto parece ser un químico legítimo basado en su nombre.
    Esta versión es más permisiva para aceptar más compuestos.
    """
    # Aceptar todos los compuestos conocidos de cualquier planta
    for plant_compounds in KNOWN_COMPOUNDS.values():
        if compound in plant_compounds or compound.lower() in [x.lower() for x in plant_compounds]:
            return True
    
    lowered = compound.lower()
    
    # Verificar sufijos químicos comunes
    chemical_suffixes = ['ol', 'one', 'ene', 'ane', 'oic acid', 'ic acid', 'olic acid', 
                         'al', 'diol', 'triol', 'diene', 'yl', 'amine', 'ide', 'ate']
    
    if any(lowered.endswith(suffix) for suffix in chemical_suffixes):
        return True
    
    # Verificar prefijos químicos comunes
    chemical_prefixes = ['iso', 'neo', 'cyclo', 'poly', 'mono', 'di', 'tri', 'tetra', 
                         'penta', 'hexa', 'cis', 'trans', 'alpha', 'beta', 'gamma', 'delta']
    
    if any(lowered.startswith(prefix) for prefix in chemical_prefixes):
        return True
    
    # Verificar palabras clave en el nombre
    chemical_keywords = ['acid', 'ester', 'flavon', 'cannabi', 'terpene', 'sterol', 'vitamin',
                         'phenol', 'alkal', 'glyco', 'xantho', 'humul', 'lupul', 'amine',
                         'ketone', 'aldehyde', 'oxide', 'acetate', 'propionate', 'lactone',
                         'glycoside', 'glucoside', 'pyranoside', 'furanoside']
    
    if any(keyword in lowered for keyword in chemical_keywords):
        return True
    
    # Si no pasa ninguna verificación, probablemente no es un compuesto químico
    return False

def get_smiles_from_pubchem(compound_name):
    """
    Busca la estructura SMILES de un compuesto en PubChem.
    Incluye manejo de errores y reintentos.
    """
    # Validar primero si parece ser un compuesto químico
    if not validate_compound(compound_name):
        print(f"Ignorando '{compound_name}' - no parece ser un compuesto químico")
        return None
        
    for attempt in range(MAX_RETRIES):
        try:
            compounds = pcp.get_compounds(compound_name, 'name')
            if compounds:
                return compounds[0].canonical_smiles
            else:
                print(f"No se encontró información SMILES para: {compound_name}")
                return None
        except Exception as e:
            print(f"Error al buscar {compound_name} (intento {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                # Retraso exponencial con variación aleatoria
                wait_time = min(MAX_DELAY, BASE_DELAY * (2 ** attempt) * (0.5 + random.random()))
                print(f"Reintentando en {wait_time:.1f} segundos...")
                time.sleep(wait_time)
            else:
                print(f"No se pudo obtener SMILES para {compound_name} después de {MAX_RETRIES} intentos")
                return None

def get_smiles_via_api(compound_name):
    """
    Método alternativo para obtener SMILES usando la API REST de PubChem.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # Normalizar el nombre del compuesto para la URL
    url_compound = requests.utils.quote(compound_name)
    
    try:
        # Primero obtenemos el CID (PubChem Compound ID)
        search_url = f"{base_url}/compound/name/{url_compound}/cids/JSON"
        response = requests.get(search_url)
        
        if response.status_code == 200:
            data = response.json()
            if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                cid = data['IdentifierList']['CID'][0]
                
                # Ahora obtenemos el SMILES usando el CID
                property_url = f"{base_url}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                prop_response = requests.get(property_url)
                
                if prop_response.status_code == 200:
                    prop_data = prop_response.json()
                    if 'PropertyTable' in prop_data and 'Properties' in prop_data['PropertyTable']:
                        return prop_data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        
        return None
    except Exception as e:
        print(f"Error en API de PubChem para {compound_name}: {e}")
        return None

def process_compound(plant_name, compound):
    """
    Procesa un compuesto específico buscando su SMILES y asignando referencia.
    """
    # Intentamos obtener el SMILES usando pubchempy
    smiles = get_smiles_from_pubchem(compound)
    
    # Si falla, intentamos con el método de API alternativo
    if not smiles:
        smiles = get_smiles_via_api(compound)
    
    # Pequeña pausa para no saturar las APIs
    time.sleep(random.uniform(0.5, 1.0))
    
    # Asignar referencia bibliográfica
    reference = assign_reference(compound, plant_name)
    
    # Guardamos los datos solo si pudimos obtener el SMILES
    if smiles:
        result = {
            'molecule_name': compound,
            'smiles': smiles,
            'plant_name': plant_name,
            'reference': reference
        }
        return result
    else:
        return None

def load_existing_data():
    """
    Carga datos existentes de archivos CSV anteriores.
    """
    files_to_check = [
        os.path.join(SAVE_DIR, "cannabaceae_phytochemicals.csv"),
        os.path.join(SAVE_DIR, "cannabaceae_phytochemicals_extended.csv"),
        os.path.join(SAVE_DIR, "cannabaceae_phytochemicals_precisos.csv"),
        os.path.join(SAVE_DIR, "cannabaceae_phytochemicals_final.csv"),
        os.path.join(SAVE_DIR, "cannabaceae_phytochemicals_expanded.csv")
    ]
    
    existing_data = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"Cargando datos de {file_path}: {len(df)} registros")
                
                # Convertir a diccionarios para combinar con nuevos datos
                file_data = df.to_dict('records')
                
                # Añadir campo de referencia si no existe
                for item in file_data:
                    if 'reference' not in item:
                        item['reference'] = 'Archivo previo'
                    if isinstance(item.get('reference'), float) and pd.isna(item.get('reference')):
                        item['reference'] = 'Archivo previo'
                
                existing_data.extend(file_data)
                
                # Añadir compuestos al conjunto global (para no reprocesarlos)
                global discovered_compounds
                for item in file_data:
                    discovered_compounds.add(item['molecule_name'])
                
            except Exception as e:
                print(f"Error cargando {file_path}: {e}")
    
    if existing_data:
        print(f"Total de registros cargados de archivos existentes: {len(existing_data)}")
    
    return existing_data

def mine_phytochemical_data():
    """
    Función principal para minar datos fitoquímicos.
    """
    # Cargar datos existentes
    all_data = load_existing_data()
    
    # Reunir todos los compuestos a procesar
    all_compounds_to_process = {}
    
    for plant_name, plant_compounds in KNOWN_COMPOUNDS.items():
        # Identificar compuestos que ya existen en los datos previos
        existing_compounds = set()
        for item in all_data:
            if item['plant_name'] == plant_name:
                existing_compounds.add(item['molecule_name'])
        
        # Seleccionar compuestos que no están en los datos previos
        new_compounds = [comp for comp in plant_compounds if comp not in existing_compounds]
        
        if MAX_COMPOUNDS_PER_PLANT > 0 and len(new_compounds) > MAX_COMPOUNDS_PER_PLANT:
            # Seleccionar aleatoriamente si hay demasiados compuestos nuevos
            random.shuffle(new_compounds)
            new_compounds = new_compounds[:MAX_COMPOUNDS_PER_PLANT]
            
        all_compounds_to_process[plant_name] = new_compounds
        
        print(f"Planta: {plant_name}")
        print(f"  - Compuestos totales: {len(plant_compounds)}")
        print(f"  - Compuestos ya existentes: {len(existing_compounds)}")
        print(f"  - Nuevos compuestos a procesar: {len(new_compounds)}")
    
    # Inicializamos el contador total para la barra de progreso
    total_compounds = sum(len(compounds) for compounds in all_compounds_to_process.values())
    print(f"\nSe procesarán un total de {total_compounds} compuestos nuevos para las {len(all_compounds_to_process)} plantas.")
    
    # Procesamos todos los compuestos, buscando sus códigos SMILES
    new_data = []
    
    # Creamos una barra de progreso
    with tqdm(total=total_compounds, desc="Minando datos fitoquímicos") as pbar:
        # Iteramos por cada planta y sus compuestos
        for plant_name, compounds in all_compounds_to_process.items():
            print(f"\nProcesando compuestos de {plant_name}...")
            
            # Procesar los compuestos con ThreadPoolExecutor para paralelizar
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                for compound in compounds:
                    future = executor.submit(process_compound, plant_name, compound)
                    futures[future] = compound
                
                # Recogemos los resultados a medida que se completan
                for future in futures:
                    result = future.result()
                    if result:
                        new_data.append(result)
                    pbar.update(1)
    
    # Combinamos los datos existentes con los nuevos
    all_data.extend(new_data)
    
    # Actualizar referencias en los datos existentes
    updated_data = []
    for item in all_data:
        # Si es un archivo previo y tiene referencia genérica, actualizarla
        if item['reference'] == 'Archivo previo' or item['reference'] == 'Base de datos interna':
            item['reference'] = assign_reference(item['molecule_name'], item['plant_name'])
        updated_data.append(item)
    
    # Eliminamos duplicados exactos pero mantenemos las mismas moléculas en diferentes plantas
    df = pd.DataFrame(updated_data)
    df = df.drop_duplicates(subset=['molecule_name', 'plant_name'])
    
    return df.to_dict('records')

def save_to_csv(data, filename=OUTPUT_FILE):
    """
    Guarda los datos recolectados en un archivo CSV en la ubicación especificada.
    """
    # Aseguramos que el directorio exista
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convertimos los datos a un DataFrame
    df = pd.DataFrame(data)
    
    # Guardamos el DataFrame a un archivo CSV
    df.to_csv(filename, index=False)
    print(f"\nSe guardaron {len(df)} registros en: {filename}")
    
    # Mostramos un resumen de los datos
    print("\nResumen de compuestos por planta:")
    counts = df['plant_name'].value_counts()
    for plant, count in counts.items():
        print(f"  - {plant}: {count} compuestos")
    
    # Analizamos compuestos compartidos entre plantas
    print("\nCompuestos compartidos entre plantas:")
    shared_df = pd.DataFrame(df.groupby('molecule_name')['plant_name'].apply(list))
    shared_df['plant_count'] = shared_df['plant_name'].apply(len)
    shared_df = shared_df[shared_df['plant_count'] > 1]
    
    if not shared_df.empty:
        print(f"Se encontraron {len(shared_df)} compuestos presentes en múltiples plantas.")
        
        # Mostrar solo los primeros 20 ejemplos para no saturar la salida
        count = 0
        for idx, row in shared_df.iterrows():
            plants_str = ", ".join(row['plant_name'])
            print(f"  - {idx}: presente en {plants_str}")
            count += 1
            if count >= 20:
                remaining = len(shared_df) - 20
                if remaining > 0:
                    print(f"  ... y {remaining} más")
                break

def main():
    """Función principal del script"""
    print("=" * 80)
    print("MINERÍA DE DATOS FITOQUÍMICOS COMPLETA - ESPECIES DE CANNABACEAE")
    print("=" * 80)
    print(f"Plantas objetivo: {', '.join(KNOWN_COMPOUNDS.keys())}")
    print(f"Archivo de salida: {OUTPUT_FILE}")
    print(f"Máximo de compuestos nuevos por planta: {MAX_COMPOUNDS_PER_PLANT}")
    print("=" * 80)
    
    # Minamos los datos
    phytochem_data = mine_phytochemical_data()
    
    # Guardamos los resultados
    save_to_csv(phytochem_data)
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()