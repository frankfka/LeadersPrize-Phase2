from experiments.util.experiment_util import get_truth_tuple_extractor

claims = [
    "The economic impact of Atlanta's 2000 Super Bowl was $292 million.",
    # Subj: I, Verb: am, Obj: product of a mixed marriage, Coordinates: "would have been illegal in 12 states when I was born"
    "I'm the product of a mixed marriage that would have been illegal in 12 states when I was born.",
    # Subj: Barack Obama, Verb: voted, Obj: against \"born alive\" legislation, Coordinates: was \"virtually identical to the federal law.\"
    "Barack Obama voted against \"born alive\" legislation in Illinois that was \"virtually identical to the federal law.\"",
    # Subj: economic stimulus, Verb: has saved or created, Obj: nearly 150,000 jobs, Coordinates: In the 100 days since its passage
    "In the 100 days since its passage, the economic stimulus has \"saved or created nearly 150,000 jobs.\"",
    # Subj: Wind, Verb: powers, Obj: nearly 13 million homes across the country, Coordinates: None
    "Wind powers nearly 13 million homes across the country.",
    # Subj: 54 percent of Latinos, Verb: were registered to vote, Obj: None, Coordinates: In 2008
    # Subj: 35 percent, Verb: turned out, Obj: None, Coordinates: In 2008
    "In 2008, \"only 54 percent of Latinos in Texas were registered to vote and only 35 percent actually turned out.\"",
    # Subj: Wisconsin's middle class, Verb: is, Obj: \"most diminished\" in the United States, Coordinates: None
    "Wisconsin\u2019s middle class is the \"most diminished\" in the United States.",
    # Subj: Democratic North Carolina House candidate Dan McCready, Verb: took, Obj:  money from the Pelosi crowd, Coordinates: None
    # Subj: Democratic North Carolina House candidate Dan McCready, Verb: is , Obj: \"with them\" in their agenda, Coordinates: None
    # Subj: Democratic North Carolina House candidate Dan McCready, Verb: wants , Obj: repeal your tax cut, Coordinates: None
    "Says Democratic North Carolina House candidate Dan McCready \"took money from the Pelosi crowd\" and is \"with them\" in their agenda and \"wants to repeal your tax cut.\"",
    "The Alternative Minimum Tax...was created by Congress in 1969 to affect 155 wealthy Americans. Because it was never indexed for inflation, those original 155 taxpayers has increased to affect about 3.5-million in 2006.\"",
    "\"John wasn't this raging populist four years ago\" when he ran for president.",
    "\"I just talked about guns. I told you what my position was, and what I did as governor, the fact that I received the endorsement of the NRA.\"",
    "We borrow money from the Chinese to buy oil from the Saudis.",
    "The musical Mamma Mia! has \"been selling out for years.\""
]


def main():
    extractor = get_truth_tuple_extractor()
    for claim in claims:
        print(claim)
        print(extractor.extract(claim))
        print("\n")


if __name__ == '__main__':
    main()
