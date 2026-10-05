"""Fresh costs contain only this run; no source16/resume accounting branch."""
def tele_tokens(records):
    prompts=[q.get('actual_engine_prompt_token_count') for q in records]
    outputs=[sum(o['output_tokens'] for o in q['outputs']) if isinstance(q.get('outputs'),list) and
             all(type(o.get('output_tokens')) is int for o in q['outputs']) else None for q in records]
    total=lambda values:sum(values) if all(type(v) is int for v in values) else None
    return {'requests':len(records),'actual_engine_prompt_tokens':total(prompts),'actual_output_tokens':total(outputs)}


def cost(records,paddle,phases,wall,synthetic,loaded=None,tele_result=None,paddle_loads=None,started_epoch=None,ended_epoch=None):
    current=tele_tokens(records);paddle_count=max([q.get('requests_so_far',0) for q in paddle] or [0])
    if not paddle and phases.get('paddle',{}).get('status')!='NOT_TRIGGERED':paddle_count=None
    paddle_loads=paddle_loads or []
    masks=[q.get('actual_attention_mask_tokens') for q in paddle]
    not_triggered=not paddle and phases.get('paddle',{}).get('status')=='NOT_TRIGGERED'
    prompt_tokens=(sum(masks) if all(type(v) is int for v in masks) else None) if paddle else (0 if not_triggered else None)
    output_tokens=0 if not_triggered else None
    model_loads={'tele':{'successful_loaded_record':loaded,'actual_model_load_count':1 if loaded else None,
                        'phase_attempt':phases.get('tele'),'peak_memory_result':tele_result},
                 'paddle':{'build_events':paddle_loads,'observed_successful_builds':sum(q.get('status')=='COMPLETE' for q in paddle_loads),
                           'phase_attempt':phases.get('paddle')}}
    incremental=dict(current,stages={s:tele_tokens([q for q in records if q.get('stage')==s]) for s in ('NATIVE','FORMULA125','GUARD','TABLE1X')},
        paddle={'requests':paddle_count,'actual_prompt_tokens':prompt_tokens,'actual_output_tokens':output_tokens,
                'output_token_status':'NOT_TRIGGERED_ZERO' if not_triggered else 'UNKNOWN; generator output token IDs not recorded'},
        model_loads=model_loads,GPU_card_seconds=0 if synthetic else None,wall_seconds=wall,attempts=phases,
        supervisor_started_epoch=started_epoch,supervisor_ended_epoch=ended_epoch,
        single_GPU_reserved_envelope_seconds=0 if synthetic else wall)
    return {'accounting_scope':'FRESH_FORMAL_RUN_THIS_RUN_ONLY','source':{'requests':0,'actual_engine_prompt_tokens':0,'actual_output_tokens':0,'GPU_card_seconds':0,'source_run':None},
            'incremental':incremental,'combined':dict(current,paddle_requests=paddle_count,GPU_card_seconds=0 if synthetic else None,wall_seconds=wall),
            'cost_status':'GPU allocation cost requires parent telemetry; unknown never zero on real run', 'synthetic':synthetic,'no_retries':True}
