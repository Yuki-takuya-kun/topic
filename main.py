import argparse


def main():
    parser = argparse.ArgumentParser(description='The commands that use to run the project')
    parser.add_argument('program', type=str, help='The Program that you want to run')
    parser.add_argument('--topic_analyze' , type=bool, default=True, help='If you want to analyze in the bertopic')
    parser.add_argument('--compute_coherence', type=bool, default=True, help='If you want to calculate the coherence with a topic model')
    parser.add_argument('--model', type=str, default='all', help='Which model to use as topic analyzer, all for all models')
    parser.add_argument('--baseline', type=str, default='all', help='Which baseline you want to run, if all then it will run all baselines')
    
    args = parser.parse_args()
    print(args.program)
    if args.program == 'topic':
        run_topic_models(args)
    elif args.program == 'run':
        run()
    else:
        raise KeyError(f"the program {args.program} do not exists")

def run_topic_models(args):
    print('run')
    import yaml
    
    from extract.lda import LDA_Analyzer
    from extract.bertopic import Bertopic_analyzer
    
    for airline in ['AirFrance','AmericanAirlines', 'Lufthansa', 'DeltaAirlines']:
        print(airline)
        #airline = 'AirChina'
        if args.model == 'all' or args.model == 'lda':
            lda_analyzer = LDA_Analyzer(airline)
            if args.topic_analyze:
                lda_analyzer.analyze()
            if args.compute_coherence:
                lda_analyzer.compute_coherence()

        if args.model == 'all' or args.model == 'bertopic':
            bertopic_analyzer = Bertopic_analyzer(airline)
            if args.topic_analyze:
                bertopic_analyzer.analyze()
            if args.compute_coherence:
                bertopic_analyzer.compute_coherence()
                
def run():
    from extract.topic import Analyzer
    import pandas as pd
    df = pd.DataFrame(columns=['airline', 'distribution', 'coherence', 'diversity'])
    for airline in ['AirChina', 'ChinaEasternAirlines', 'AmericanAirlines','DeltaAirlines','AirFrance', 'Lufthansa']:
    #for airline in ['AmericanAirlines']:
        print(f'using airline {airline}')
        analyzer = Analyzer(airline)
        res = analyzer.analyze()
        print(res)
        for r in res:
            df.loc[len(df)] = r
    print(df)
    df.to_csv('output/results.csv', index=False)

if __name__ == '__main__':       
    main()  