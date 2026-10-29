"""Observed-time reconstruction must never use a future incumbent."""
import unittest
import numpy as np
import pandas as pd

from paper_a_report import prepare_paper_data
from paper_clock_report import reconstruct_clock_quality, clock_summaries
from test_paper_experiment_a import fixture


def clock_fixture():
    raw,manifest=fixture('pilot')
    data=prepare_paper_data(raw,manifest,True)
    traces=[]
    for _,run in data['benchmark'][data['benchmark']['Paper method'].isin(['Cold','GBStart','GDStart'])].iterrows():
        m=run['Paper method'];initial=0. if m=='Cold' else run['Warm Start Value']
        base={'Scenario':run['Scenario'],'Tasks':run['Tasks'],'Seed':run['Seed'],'Start Method':m,
              'Budget sec':600.,'Objective SHA256':data['spec'].fingerprint,'Study Stage':'Warm'}
        for time,event,solver,available in [
                (0.,'Idle fallback available' if m=='Cold' else 'Validated start available',np.nan,initial),
                (2.,'HiGHS improving incumbent',.8*run['Certified optimum'],max(initial,.8*run['Certified optimum'])),
                (run['Solver Time sec'],'Solve finished',run['Certified optimum'],run['Certified optimum'])]:
            traces.append({**base,'Elapsed solve wall sec':time,'Event':event,
                           'Solver Incumbent Value':solver,'Available Value':available})
    return data,pd.DataFrame(traces)


class ObservedClockTests(unittest.TestCase):
    def test_future_incumbents_and_certificates_are_not_backfilled(self):
        data,traces=clock_fixture()
        detail=reconstruct_clock_quality(data,traces)
        first=detail[detail['Elapsed solve wall sec'].eq(1)].set_index('Start Method')
        self.assertFalse(first.loc['Cold','Available plan'])
        self.assertEqual(first.loc['Cold','Available value'],0.)
        self.assertTrue(pd.isna(first.loc['Cold','Observed solver incumbent']))
        self.assertTrue(first.loc['GBStart','Available plan'])
        self.assertFalse(first['Observed solver plan'].any())
        five=detail[detail['Elapsed solve wall sec'].eq(5)]
        self.assertTrue(np.allclose(five['Available shortfall pct'],20.))
        self.assertFalse(five['Certified by time'].any())
        fifteen=detail[detail['Elapsed solve wall sec'].eq(15)]
        self.assertTrue(fifteen['Certified by time'].all())
        self.assertTrue(np.allclose(fifteen['Available shortfall pct'],0.))
        for _,group in detail.groupby('Start Method'):
            self.assertTrue(group['Available value'].diff().dropna().ge(-1e-10).all())

    def test_missing_history_and_unknown_reference_are_explicit(self):
        data,traces=clock_fixture()
        with self.assertRaisesRegex(ValueError,'Incomplete initial/final history'):
            reconstruct_clock_quality(data,traces[traces['Event'].ne('Solve finished')])
        data['benchmark']['Certified optimum']=np.nan
        detail=reconstruct_clock_quality(data,traces)
        summary=clock_summaries(detail)
        self.assertTrue(summary['known_references'].eq(0).all())
        self.assertTrue(summary['mean_available_shortfall_pct'].isna().all())
        self.assertTrue(summary['mean_available_value'].notna().all())


if __name__=='__main__':unittest.main()
