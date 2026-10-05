"""Terminal denominator and actual four-stage completion; never a score."""
import math


def summarize(rows,terminals,stages,pages,audits,phase_failed,synthetic):
    ids=[row['id'] for row in rows];terminal_ids=[t['id'] for t in terminals]
    roster_ok=terminal_ids==ids and len(set(terminal_ids))==len(ids) and all(t.get('in_denominator') is True for t in terminals)
    expected={(b,s) for b in range(math.ceil(len(rows)/16)) for s in ('NATIVE','FORMULA125','GUARD','TABLE1X')}
    completed={(v.get('batch'),v.get('stage')) for v in stages if v.get('status')=='COMPLETE'}
    stage_ok=completed==expected
    page_ids=[row['id'].rsplit('.',1)[0] for row in rows]
    completed_pages=[p.get('page_id') for p in pages if p.get('status')=='COMPLETE']
    pages_ok=completed_pages==page_ids and len(set(completed_pages))==len(page_ids)
    audit_ok=len(audits)>=2 and all(a.get('status')=='PASS' for a in audits)
    failures=sum(t.get('state')!='COMPLETE' for t in terminals)
    return {'terminal_roster_complete':roster_ok,'terminal_pages':len(terminals),'original_denominator_pages':len(rows),
            'complete_pages':len(terminals)-failures,'failed_pages_retained':failures,
            'all_four_stages_complete':stage_ok,'actual_page_completion_ledger_matches':pages_ok,
            'loaded_source_audits_pass':audit_ok,'protocol_completed':not synthetic and not phase_failed and
                roster_ok and stage_ok and pages_ok and audit_ok and failures==0,
            'cold_start_smoke_pass':False,'score':None}
