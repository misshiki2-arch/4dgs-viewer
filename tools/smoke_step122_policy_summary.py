"""Versioned observer regression; old gates never authorize the new path."""
import copy
import unittest
import summarize_step_json as summary


class PolicySummary(unittest.TestCase):
    def policy(self, mode='NEW-ON', count=0):
        return dict(schema='adopted-range-compositing-policy-v1',mathVersion='cuda-aligned-composite-v1',
            appliedMode=mode,requestedMode=mode,purpose='validation-only',executed=True,
            onPathExecuted=mode=='NEW-ON',terminationObserved=count>0,terminatedPixelCount=count,
            normalOnAccepted=False,accepted=False,offOnComparisonComplete=False,
            evaluatedPixelReferenceCount=None,skippedPixelReferenceCount=None)

    def test_versioned_consumer_and_legacy_separation(self):
        legacy={'webgpuTileListCompositorContract':{'earlyTerminationEnabled':False,
                'earlyTerminationDisabledBySafetyGate':True,'earlyTerminationRemainsDisabled':True}}
        old=summary.build_step104_compositor_work_reduction_visual_safety_summary(legacy)
        self.assertEqual(summary.build_step122_compositing_policy_summary(legacy)['status'],'not-recorded')
        for mode,count in [('NEW-OFF',0),('NEW-ON',0),('NEW-ON',1)]:
            value=copy.deepcopy(legacy)
            value['webgpuTileListCompositorContract'].update(compositingPolicy=self.policy(mode,count),
                compositingPolicyCapability=dict(version='adopted-range-compositing-capability-v1',
                    executionReady=True,normalOnAccepted=False,comparisonComplete=False))
            result=summary.extract_webgpu_visible_record_dryrun_v1({'webgpuVisibleRecordDryRun':value})
            observed=result['step122CompositingPolicyObservation']
            self.assertTrue(observed['executionReady']);self.assertFalse(observed['accepted'])
            self.assertFalse(observed['normalOnAccepted']);self.assertFalse(observed['offOnComparisonComplete'])
            self.assertEqual(observed['terminationObserved'],count>0)
            self.assertEqual(summary.build_step104_compositor_work_reduction_visual_safety_summary(value),old)
            for key,bad in [('schema','unknown'),('mathVersion','old'),('appliedMode','OLD'),
                            ('requestedMode','OLD'),('purpose','normal' if mode=='NEW-ON' else 'unknown'),
                            ('terminatedPixelCount',True),('accepted',True),('evaluatedPixelReferenceCount',4)]:
                altered=copy.deepcopy(value);altered['webgpuTileListCompositorContract']['compositingPolicy'][key]=bad
                with self.subTest(mode=mode,key=key):
                    got=summary.build_step122_compositing_policy_summary(altered)
                    self.assertEqual(got['status'],'blocked');self.assertFalse(got['executionReady'])
            value['webgpuTileListCompositorContract']['compositingPolicyCapability']=None
            self.assertFalse(summary.build_step122_compositing_policy_summary(value)['executionReady'])


if __name__=='__main__':unittest.main(verbosity=2)
