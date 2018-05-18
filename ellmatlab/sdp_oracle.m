classdef sdp_oracle < handle
    properties
        c, lmi
	end
	methods
		function obj = sdp_oracle(c, F)
			obj.c = c;
			obj.lmi = lmi_oracle(F);
        end

		function [g, fj), t] = evaluate(obj, x, t)
            f0 = obj.c'*x;
            fj = f0 - t;
            if fj > 0, g = obj.c; return; end	

            [g, fj] = obj.lmi.chk_spd(x);
            if g ~= 0, return; end;
			
			t = f0; fj = 0; end			
        end
	end
end

