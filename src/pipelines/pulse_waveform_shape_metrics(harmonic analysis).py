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
    v_raw = "/Artery/VelocityPerBeat/VelocitySignalPerBeat/value"
    v = "/Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    T = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    def run(self, h5file) -> ProcessResult:
        v_raw = np.asarray(h5file[self.v])
        v_ds = np.asarray(h5file[self.v])
        t_ds = np.asarray(h5file[self.T])
        
        N=len(v_ds[:,0])
        nb_harmonic=10
        Xn=[]
        
        CV_1=[]
        CV_2=[]
        Vn_tot=[]
        
        sigma_phase_tot=[]
        v_hat=[]
        CF_tot=[]
        D_alpha=[]
        V0=[]
        FVTI=[]
        IVTI=[]
        max=[]
        
        
        alpha=0.5 #that needs to be specified
        for k in range(len(v_ds[0])):
            
            fft_vals=np.fft.fft(v_ds[:,k])/N
            limit=nb_harmonic+1
            Vn=fft_vals[1:limit]
            V1=Vn[0]
            Xn_k=Vn/V1
            Xn.append(Xn_k)
            Vn_tot.append(Vn)
            
            omega0= (2 * np.pi)/t_ds[0][k]
            V0.append(np.mean(v_ds[:,k]))
            
            v_hat_k=np.zeros_like(t_ds[0][k])
            
            for n, v_coeff in enumerate (Vn,1):
                h=n*omega0*2*(v_coeff*np.exp(1j*n*omega0*t_ds[0][k])).real
                v_hat_k+=h
            v_hat_V0=v_hat_k+V0[k]
            v_hat.append(v_hat_V0)

            v_hat_np=np.asarray(v_hat)
            diff=v_hat_np.T[k]-v_ds[:,k]
            max.append(np.maximum(diff, 0))
            max_np=np.asarray(max)

            RMST=np.sqrt(np.mean(v_hat[k]**2)) 
            CF=np.max(v_hat)/RMST
            CF_tot.append(CF)
           
            amp1=np.abs(np.asarray(Vn_tot)[:,0])
            amp2=np.abs(np.asarray(Vn_tot)[:,1])

            CV_1.append(np.std(amp1)/np.mean(amp1))
            CV_2.append(np.std(amp2)/np.mean(amp2))

            phi1=np.angle(np.asarray(Xn)[:,0])
            phi2=np.angle(np.asarray(Xn)[:,1]) 

            diff_phase=phi2-2*phi1
            diff_phase_wrap=(diff_phase+np.pi)%(2*np.pi)-np.pi
            sigma_phase=np.std(diff_phase_wrap)
            sigma_phase_tot.append(sigma_phase)

            seuil=V0[k]+alpha*np.std(v_hat[k])
            condition= v_hat[k] > seuil 
            D_alpha.append(np.mean(condition))

                   
            moitie = len(max) // 2
            d1 = np.sum(max_np.T[k][:moitie])
            d2 = np.sum(max_np.T[k][moitie:])
            AVTI=np.sum(max_np.T[k])
            FVTI.append(d1/AVTI)
            IVTI.append((d1-d2)/(d1+d2))

        
        metrics = {
            "Xn": with_attrs(
                np.asarray(Xn),
                {
                    "unit": [""],
                },
            ),
            "Beat to beat amplitude stability (n=1)": with_attrs(
                np.asarray(CV_1),
                {
                    "unit": [""],
                },
            ),
            "Beat to beat amplitude stability (n=2)": with_attrs(
                np.asarray(CV_2),
                {
                    "unit": [""],
                },
            ),
            "Beat to beat phase coupling stability (n=2)": with_attrs(
                np.asarray(sigma_phase_tot),
                {
                    "unit": [""],
                },
            ),
             "Band limited waveform (definition) : v_hat": with_attrs(
                np.asarray(v_hat),
                {
                    "unit": [""],
                },
            ), 
            "Band limited crest factor : CF": with_attrs(
                np.asarray(CF_tot),
                {
                    "unit": [""],
                },
            ), 
            "Effective Duty cycle : D_alpha": with_attrs(
                np.asarray(D_alpha),
                {
                    "unit": [""],
                },
            ), 
            "V0": with_attrs(
                np.asarray(V0),
                {
                    "unit": [""],
                },
            ), 
            "Normalised first half fraction : FVTI": with_attrs(
                np.asarray(FVTI),
                {
                    "unit": [""],
                },
            ), 
            
            "VTI asymmetry index : IVTI": with_attrs(
                np.asarray(IVTI),
                {
                    "unit": [""],
                },
            ),         
            

            
        }
        # Artifacts can store non-metric outputs (strings, paths, etc.).

        return ProcessResult(metrics=metrics)