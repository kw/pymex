from matlab import mx
import sys

def _findtype(typelist):
    for (modstr, clsstr) in typelist:
        if len(modstr): 
            adjusted_modstr = "mltypes.%s" % modstr
        else:
            adjusted_modstr = "mltypes"
            try:
                tempmod = __import__(adjusted_modstr, globals(), locals(), [clsstr], 0)
                cls = getattr(tempmod, clsstr);
                return cls;
            except:
                pass            
    # if we got this far and found nothing, just use mx.Array
    return mx.Array;


class cell(mx.Array): pass
