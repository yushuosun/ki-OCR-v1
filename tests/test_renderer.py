"""Reject the observed official CDM silent-render failure and ineffective controls."""
import copy, unittest
from renderer_smoke import controls_pass

class RendererControls(unittest.TestCase):
    def setUp(self):
        self.records=[{'metrics':{'F1_score':score,'gt_tokens':5,'pred_tokens':5},
                       'artifacts':{str(i):'synthetic' for i in range(6)}} for score in (1,.286,1)]
    def test_effective_synthetic_controls(self):
        self.assertTrue(controls_pass(self.records))
    def test_silent_empty_negative_render_is_not_success(self):
        self.records[1]['metrics'].update(F1_score=0,pred_tokens=0)
        self.assertFalse(controls_pass(self.records))
    def test_ineffective_negative_control_is_rejected(self):
        self.records[1]['metrics']['F1_score']=1
        self.assertFalse(controls_pass(self.records))
    def test_missing_cjk_png_is_rejected(self):
        self.records[2]['artifacts']={}
        self.assertFalse(controls_pass(self.records))

if __name__=='__main__':unittest.main()
