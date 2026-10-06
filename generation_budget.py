"""One total generation-item ledger shared by Tele and Paddle phases, no retry."""
import fcntl
import json
import os
from pathlib import Path
import time
import operator

def paddle_items(args, options):
    # Pinned paddlex3.7.2 actual call: model.generate(inputs_dict, **kwargs).
    if args:
        if len(args)!=1 or not isinstance(args[0],dict) or 'inputs' in options or 'input_ids' in options:
            raise ValueError('Ambiguous Paddle input cardinality')
        inputs=args[0]
    elif 'inputs' in options:
        if not isinstance(options['inputs'],dict) or 'input_ids' in options:
            raise ValueError('Ambiguous Paddle input cardinality')
        inputs=options['inputs']
    else:raise ValueError('Unknown Paddle input cardinality')
    ids=inputs.get('input_ids')
    shape=getattr(ids,'shape',None)
    try:
        if shape is None or len(shape)!=2 or any(isinstance(x,bool) for x in shape):raise ValueError()
        batch,length=(operator.index(x) for x in shape)
        if batch<1 or length<1:raise ValueError()
    except (TypeError,ValueError):raise ValueError('Unknown Paddle input cardinality')
    return batch

class GenerationBudget:
    def __init__(self, root, maximum):
        self.path=Path(root)/'generation-budget.json'
        self.maximum=maximum
        if type(maximum) is not int or maximum<1:raise ValueError('Positive request maximum required')

    def initialize(self):
        with self.path.open('x',encoding='utf-8') as f:
            json.dump({'maximum':self.maximum,'reserved_items':0,'calls':[]},f)

    def reserve(self, count, phase):
        if type(count) is not int or count<1:raise ValueError('Unknown generation cardinality')
        with self.path.open('r+',encoding='utf-8') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            x=json.load(f)
            if x['maximum']!=self.maximum or x['reserved_items']+count>self.maximum:
                raise RuntimeError('GENERATION_BUDGET_EXHAUSTED_BEFORE_SUBMISSION')
            x['reserved_items']+=count
            x['calls'].append({'phase':phase,'items':count,'reserved_epoch':time.time(),'pid':os.getpid()})
            f.seek(0);json.dump(x,f);f.truncate();f.flush();os.fsync(f.fileno())
        return x['reserved_items']

def hook_tele(llm, budget, eager, observe):
    init=llm.__init__;generate=llm.generate
    def wrapped_init(self,*args,**kwargs):
        kwargs['enforce_eager']=eager
        init(self,*args,**kwargs)
        config=self.llm_engine.vllm_config.model_config
        if config.enforce_eager is not eager or config.seed!=0:raise ValueError('Actual eager/seed mismatch')
        observe({'requested_enforce_eager':eager,'actual_enforce_eager':config.enforce_eager,'seed':config.seed})
    def wrapped_generate(self,*args,**kwargs):
        prompts=args[0] if args else kwargs.get('prompts')
        if prompts is None:raise ValueError('Unknown prompt cardinality')
        budget.reserve(len(prompts) if isinstance(prompts,list) else 1,'tele')
        return generate(self,*args,**kwargs)
    llm.__init__=wrapped_init;llm.generate=wrapped_generate

def hook_paddle(predictor, budget):
    build=predictor._build
    def wrapped_build(self,**kwargs):
        model,processor=build(self,**kwargs);generate=model.generate
        def wrapped_generate(*args,**options):
            budget.reserve(paddle_items(args,options),'paddle')
            return generate(*args,**options)
        model.generate=wrapped_generate
        return model,processor
    predictor._build=wrapped_build
