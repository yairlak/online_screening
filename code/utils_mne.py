
def create_metadata(TrialInfo):
    import pandas as pd
    for k in TrialInfo.keys():
        print(k, len(TrialInfo[k]), TrialInfo[k])
    del TrialInfo['word_strings']
    return pd.DataFrame(TrialInfo)
