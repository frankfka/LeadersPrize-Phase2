from reasoners.ensemble_classifier import EnsembleClassifier

hypotheses = [
    "Bloggers Says a Michigan funeral home employee was \'cremated by mistake while taking a nap.\'",
    "When Fidel Castro came into office, you know what he did? He had a massive literacy program",
    "America’s carbon emissions are being reduced at a rate far higher than any country in the world",
    "Coronavirus is spreading faster than SARS, Ebola, and Swine Flu"
]
supports = [
    [
        "Back in 2017 we debunked a story that a Texas morgue employee was accidentally cremated while taking a nap.",
        "Now a Michigan funeral home employee was allegedly \"cremated by mistake while taking a nap.\"",
        "Maybe it\u2019s gallow\u2019s humor.",
        "The Michigan state McDonald's employee was grilling burgers but fell asleep",
        "But the Feb. 25 post was flagged as part of Facebook\u2019s efforts to combat false news and misinformation on its News Feed.",
        "(Read more about our partnership with Facebook.)",
        "That\u2019s because it seems the story originated on this website, which describes itself as a \"satire web publication.\" "
    ],
    [
        "Sanders\u2019 focus on the teaching side of Castro\u2019s campaign downplayed its substantial political overtones.",
        "But Castro did significantly expand literacy rates after seizing power.",
        "Fidel Castro was a brutal dictator in office that invested heavily in the military",
        "The academic consensus seems to be that Fidel Castro\u2019s government did increase the literacy rate of the island\u2019s population, at the same time that it clearly used the literacy campaign for propaganda purposes,\" said Jorge Duany, director of the Cuban Research Institute at Florida International University.",
        "During the year-long, massive national effort that was to be the result of Castro's daringdeclaration, 707,212 people became literate, or achieved a level of reading and writing equivalentto that of a first-grader. ",
        "Cuba's overall illiteracy rate was reduced from over 20 percent, according to the last census taken before the Revolution, to 3.9 percent, a rate far lower than thatof any other Latin American country"
    ],
    [
        "The United States saw the largest decline in energy-related CO2 emissions in 2019 on a country basis – a fall of 140 Mt, or 2.9%, to 4.8 Gt.",
        "US emissions are now down almost 1 Gt from their peak in the year 2000",
        "Tesla's new vehicle should help battle rising emission levels in the US",
        "The US is one of the best producers of diabetes than any other country",
        "But the IEA did not say the U.S. is cutting emission higher rate than any nation, let alone a \"far-higher\" rate that Gilbert described.",
        "America’s 2.9% percent reduction last year was below Japan’s, Germany’s, and the combined rate of European Union nations."
    ],
    [
        "However, the video that went viral was not the full video but a clip.",
        "Ebola was a highly contagious virus that was very prevalent in Africa",
        "The longer version clearly shows that, although the coronavirus has spread quickly, it is nowhere near as widespread as previous epidemics such as swine flu or the Spanish flu.",
        "While the coronavirus is rightly causing concern around the globe, WHO has attempted to quell some of those fears. ",
        "On a web page dedicated to information about the outbreak, WHO wrote on Feb. 23 that if you don’t live in, or have traveled recently to, an area where coronavirus is spreading, your chances of getting the disease are currently low"
    ]
]

if __name__ == '__main__':
    classifier = EnsembleClassifier()

    for hypothesis_i, hypothesis in enumerate(hypotheses):
        hypothesis_name = f'{hypothesis_i} {hypothesis}'

        sentences = supports[hypothesis_i]
        relevant_sent_idxs, preds = classifier.predict_statement_in_contexts(statement=hypothesis, contexts=sentences)
        print("\n\n")
        print(hypothesis_name)
        for pred_idx, sent_idx in enumerate(relevant_sent_idxs):
            print("\n")
            print(sentences[sent_idx])
            print(preds[pred_idx])
