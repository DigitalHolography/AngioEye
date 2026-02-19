import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="Harmonic metrics")
class ArterialHarmonicAnalysis(ProcessPipeline):
    """
    Tutorial pipeline showing the full surface area of a pipeline:

    - Subclass ProcessPipeline and implement `run(self, h5file) -> ProcessResult`.
    - Return metrics (scalars, vectors, matrices, cubes) and optional artifacts.
    - Attach HDF5 attributes to any metric via `with_attrs(data, attrs_dict)`.
    - Add attributes to the pipeline group (`attrs`) or root file (`file_attrs`).
    - No input data is required; this pipeline is purely illustrative.
    """

    description = "Tutorial: metrics + artifacts + dataset attrs + file/pipeline attrs."
    v_raw_input = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v_bandlimited_input = "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    T = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    def run(self, h5file) -> ProcessResult:
        v_raw = np.asarray(h5file[self.v_raw_input])
        v_bandlimited = np.asarray(h5file[self.v_raw_input])
        beat_period = np.asarray(h5file[self.T])
        

        
        N=len(v_bandlimited[:,0])
        nb_harmonic=10
        Xn=[]
        
        CV_1=[]
        CV_2=[]
        
        Vn_tot=[]
        sigma_phase_tot=[]
        v_hat=[]
        CF_tot=[]
        D_alpha=[]
        
        FVTI=[]
        IVTI=[]
        AVTI_tot=[]
        
        
        alpha=0.5 #that needs to be specified
        for beat_idx in range(len(v_bandlimited[0])):
            T=beat_period[0][beat_idx]
            fft_vals=np.fft.fft(v_bandlimited[:,beat_idx])/N
            limit=nb_harmonic+1
            Vn=fft_vals[:limit]
            V1=Vn[1]
            Xn_k=Vn/V1
            Xn.append(Xn_k)
            Vn_tot.append(Vn)
            
            omega0= (2 * np.pi)/beat_period
            time=np.linspace(0,T,N)
            
            v_hat_k=np.real(Vn[0])*np.ones_like(time)
            
            for i in range (1,limit):
                h=2*(Vn[i]*np.exp(1j*i*omega0*time)).real
                v_hat_k+=h
            v_hat.append(v_hat_k)

            RMST=np.sqrt(np.mean(v_hat[beat_idx]**2)) 
            CF=np.max(v_hat_k)/RMST
            CF_tot.append(CF)
           
            amp1=np.abs(np.asarray(Vn_tot)[:,1])
            amp2=np.abs(np.asarray(Vn_tot)[:,2])

            CV_1.append(np.std(amp1)/np.mean(amp1))
            CV_2.append(np.std(amp2)/np.mean(amp2))

            phi1=np.angle(np.asarray(Xn)[:,1])
            phi2=np.angle(np.asarray(Xn)[:,2]) 

            diff_phase=phi2-2*phi1
            diff_phase_wrap=(diff_phase+np.pi)%(2*np.pi)-np.pi
            sigma_phase=np.std(diff_phase_wrap)
            sigma_phase_tot.append(sigma_phase)

            seuil=Vn[0]+alpha*np.std(v_hat)
            condition= v_hat > seuil 
            D_alpha.append(np.mean(condition))

            v_base=np.min(v_hat[beat_idx])
            max=np.maximum(v_hat[beat_idx]-v_base, 0)
            
            dt=time[1]-time[0]
            moitie = len(time) // 2
            d1 = np.sum(max[:moitie])*dt
            d2 = np.sum(max[moitie:])*dt
            AVTI=np.sum(max)*dt
            AVTI_tot.append(AVTI)
            FVTI.append(d1/AVTI)
            IVTI.append((d1-d2)/(d1+d2))

        
        metrics = {
            "Xn": with_attrs(
                np.asarray(Xn),
                {
                    "unit": [""],
                },
            ),
            "AVTI": with_attrs(
                np.asarray(AVTI_tot),
                {
                    "unit": [""],
                },
            ),
            "CV1 : Beat to beat amplitude stability (n=1)": with_attrs(
                np.asarray(CV_1),
                {
                    "unit": [""],
                },
            ),
            "CV2 : Beat to beat amplitude stability (n=2)": with_attrs(
                np.asarray(CV_2),
                {
                    "unit": [""],
                },
            ),
            "sigma : Beat to beat phase coupling stability (n=2)": with_attrs(
                np.asarray(sigma_phase_tot),
                {
                    "unit": [""],
                },
            ),
             "v_hat : Band limited waveform (definition)": with_attrs(
                np.asarray(v_hat),
                {
                    "unit": [""],
                },
            ), 
            "CF : Band limited crest factor ": with_attrs(
                np.asarray(CF_tot),
                {
                    "unit": [""],
                },
            ), 
            " D_alpha : Effective Duty cycle ": with_attrs(
                np.asarray(D_alpha),
                {
                    "unit": [""],
                },
            ),
            "FVTI : Normalised first half fraction ": with_attrs(
                np.asarray(FVTI),
                {
                    "unit": [""],
                },
            ), 
            
            "IVTI : VTI asymmetry index ": with_attrs(
                np.asarray(IVTI),
                {
                    "unit": [""],
                },
            ),         
            

            
        }
        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)