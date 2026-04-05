import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="waveform_harmonic_organization")
class WaveformHarmonicOrganization(ProcessPipeline):
    """
    Direct harmonic-organization metrics on beat-resolved segment waveforms.

    This pipeline implements the direct statistics described in the internship
    proposal on normalized complex harmonic coefficients c_hbkr = V_hbkr / V_1bkr,
    excluding low-rank matrix/tensor decompositions.

    Inputs
    ------
    - raw per-segment arterial waveforms:
        /Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value
    - raw per-segment venous waveforms:
        /Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value
    - beat periods:
        /Artery/VelocityPerBeat/beatPeriodSeconds/value

    Expected segment layout
    -----------------------
    v_block[t, beat, branch, radius]
    """

    description = (
        "Direct harmonic-organization metrics on per-segment beat-resolved arterial "
        "and venous waveforms: normalized complex harmonics, coherence, phase locking, "
        "spread, occupancy, and summary descriptors."
    )

    # ----------------------------
    # Inputs
    # ----------------------------
    v_raw_segment_input_artery = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_raw_segment_input_vein = (
        "/Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    # ----------------------------
    # Parameters
    # ----------------------------
    eps = 1e-12
    H_MAX = 10
    fundamental_abs_threshold = 1e-12

    # ----------------------------
    # Small dtype helpers
    # ----------------------------
    @staticmethod
    def _f32(x):
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _i32(x):
        return np.asarray(x, dtype=np.int32)

    @staticmethod
    def _u8(x):
        return np.asarray(x, dtype=np.uint8)

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _safe_nanmedian(x: np.ndarray, axis=None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return np.nan
        with np.errstate(all="ignore"):
            return np.nanmedian(x, axis=axis)

    @staticmethod
    def _safe_nanstd(x: np.ndarray, axis=None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return np.nan
        with np.errstate(all="ignore"):
            return np.nanstd(x, axis=axis)

    @staticmethod
    def _ensure_beat_periods(T: np.ndarray, n_beats: int) -> np.ndarray:
        """
        Return T as shape (n_beats,).
        Accepts common layouts such as (1, n_beats), (n_beats, 1), or (n_beats,).
        """
        T = np.asarray(T, dtype=float)

        if T.ndim == 1:
            if T.size != n_beats:
                raise ValueError(
                    f"Beat-period vector length mismatch: got {T.size}, expected {n_beats}"
                )
            return T

        if T.ndim == 2:
            if T.shape == (1, n_beats):
                return T[0]
            if T.shape == (n_beats, 1):
                return T[:, 0]
            if T.shape[0] == 1 and T.shape[1] >= n_beats:
                return T[0, :n_beats]
            if T.shape[1] == 1 and T.shape[0] >= n_beats:
                return T[:n_beats, 0]

        raise ValueError(f"Could not interpret beat periods with shape {T.shape}")

    @staticmethod
    def _complex_nan(shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)

    @staticmethod
    def _isfinite_complex(x: np.ndarray) -> np.ndarray:
        return np.isfinite(np.real(x)) & np.isfinite(np.imag(x))

    def _complex_mean(self, x: np.ndarray, axis) -> np.ndarray:
        """
        Mean of a complex array ignoring NaN+1j*NaN entries.
        """
        x = np.asarray(x, dtype=np.complex128)
        mask = self._isfinite_complex(x)
        sumx = np.sum(np.where(mask, x, 0.0 + 0.0j), axis=axis)
        count = np.sum(mask, axis=axis)

        if np.ndim(sumx) == 0:
            if count > 0:
                return np.complex128(sumx / count)
            return np.complex128(np.nan + 1j * np.nan)

        out = self._complex_nan(sumx.shape)
        valid = count > 0
        out[valid] = sumx[valid] / count[valid]
        return out

    def _complex_resultant(self, angles: np.ndarray, axis) -> np.ndarray:
        """
        Circular mean resultant of phase angles, ignoring NaNs.
        """
        angles = np.asarray(angles, dtype=float)
        valid = np.isfinite(angles)
        z = np.where(valid, np.exp(1j * angles), np.nan + 1j * np.nan)
        return self._complex_mean(z, axis=axis)

    def _entropy_effective_number(self, weights: np.ndarray, axis: int) -> np.ndarray:
        """
        Effective number exp(-sum p log p) along axis, with p normalized along axis.
        """
        w = np.asarray(weights, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        s = np.sum(w, axis=axis, keepdims=True)
        p = np.divide(w, s, out=np.zeros_like(w), where=s > 0)

        positive = p > 0
        term = np.zeros_like(p, dtype=float)
        term[positive] = p[positive] * np.log(p[positive])
        H = -np.sum(term, axis=axis)
        N_eff = np.exp(H)

        total = np.squeeze(s, axis=axis)

        if np.ndim(N_eff) == 0:
            return float(N_eff) if total > 0 else np.nan

        out = np.full_like(N_eff, np.nan, dtype=float)
        valid = total > 0
        out[valid] = N_eff[valid]
        return out

    # ----------------------------
    # Harmonics
    # ----------------------------
    def _harmonic_coefficients_block(
        self, v_block: np.ndarray, T_vec: np.ndarray
    ) -> dict:
        """
        Compute beat/segment harmonic coefficients V_hbkr from FFT of each 1-beat waveform.

        Returns
        -------
        dict with:
        - V: complex array (H+1, B, K, R), harmonics 0..H
        - valid_waveform_mask: bool array (B, K, R)
        - H: int
        """
        if v_block.ndim != 4:
            raise ValueError(
                f"Expected segment block shape (n_t,n_beats,n_branches,n_radii), "
                f"got {v_block.shape}"
            )

        v_block = np.asarray(v_block, dtype=float)
        n_t, n_beats, n_branches, n_radii = v_block.shape
        T_vec = self._ensure_beat_periods(T_vec, n_beats)

        H = int(min(self.H_MAX, n_t // 2))
        if H < 1:
            raise ValueError(f"Need at least one harmonic, got H={H} for n_t={n_t}")

        V = self._complex_nan((H + 1, n_beats, n_branches, n_radii))
        valid_waveform_mask = np.zeros((n_beats, n_branches, n_radii), dtype=bool)

        for b in range(n_beats):
            Tbeat = float(T_vec[b])
            if (not np.isfinite(Tbeat)) or Tbeat <= 0:
                continue

            for k in range(n_branches):
                for r in range(n_radii):
                    v = v_block[:, b, k, r]
                    if v.size < 2 or not np.all(np.isfinite(v)):
                        continue

                    Vf = np.fft.rfft(v) / float(v.size)
                    if Vf.size < H + 1:
                        continue

                    V[:, b, k, r] = Vf[: H + 1]
                    valid_waveform_mask[b, k, r] = True

        return {"V": V, "H": H, "valid_waveform_mask": valid_waveform_mask}

    def _normalized_harmonics(self, V: np.ndarray) -> dict:
        """
        Compute c_hbkr = V_hbkr / V_1bkr for h=2..H, whenever |V1| is sufficiently large.
        """
        H = int(V.shape[0] - 1)
        n_beats, n_branches, n_radii = V.shape[1:]

        if H < 2:
            c = self._complex_nan((0, n_beats, n_branches, n_radii))
            stable = np.zeros((0, n_beats, n_branches, n_radii), dtype=bool)
            harmonics = np.asarray([], dtype=int)
            return {"c": c, "stable_mask": stable, "harmonics": harmonics}

        V1 = V[1]
        stable_fundamental = self._isfinite_complex(V1) & (
            np.abs(V1) > float(self.fundamental_abs_threshold)
        )

        c = self._complex_nan((H - 1, n_beats, n_branches, n_radii))
        stable_mask = np.zeros((H - 1, n_beats, n_branches, n_radii), dtype=bool)

        for hi, h in enumerate(range(2, H + 1)):
            Vh = V[h]
            ok = stable_fundamental & self._isfinite_complex(Vh)
            stable_mask[hi] = ok

            c_h = self._complex_nan((n_beats, n_branches, n_radii))
            c_h[ok] = Vh[ok] / V1[ok]
            c[hi] = c_h

        harmonics = np.arange(2, H + 1, dtype=int)
        return {"c": c, "stable_mask": stable_mask, "harmonics": harmonics}

    # ----------------------------
    # Direct metrics
    # ----------------------------
    def _beat_and_location_means(self, c: np.ndarray) -> dict:
        cbar_b_hkr = self._complex_mean(c, axis=1)      # (H-1, K, R)
        cbar_kr_hb = self._complex_mean(c, axis=(2, 3)) # (H-1, B)
        return {"cbar_b_hkr": cbar_b_hkr, "cbar_kr_hb": cbar_kr_hb}

    def _coherence_metrics(self, c: np.ndarray) -> dict:
        valid = self._isfinite_complex(c)
        abs_c = np.where(valid, np.abs(c), 0.0)

        sum_b = np.sum(np.where(valid, c, 0.0 + 0.0j), axis=1)
        den_b = np.sum(abs_c, axis=1)
        Gamma_b_hkr = np.full(den_b.shape, np.nan, dtype=float)
        ok_b = den_b > 0
        Gamma_b_hkr[ok_b] = np.abs(sum_b[ok_b]) / den_b[ok_b]

        sum_kr = np.sum(np.where(valid, c, 0.0 + 0.0j), axis=(2, 3))
        den_kr = np.sum(abs_c, axis=(2, 3))
        Gamma_kr_hb = np.full(den_kr.shape, np.nan, dtype=float)
        ok_kr = den_kr > 0
        Gamma_kr_hb[ok_kr] = np.abs(sum_kr[ok_kr]) / den_kr[ok_kr]

        return {"Gamma_b_hkr": Gamma_b_hkr, "Gamma_kr_hb": Gamma_kr_hb}

    def _low_order_phase_metrics(self, c: np.ndarray, harmonics: np.ndarray) -> dict:
        keep_mask = np.isin(harmonics, np.asarray([2, 3], dtype=int))
        low_harmonics = harmonics[keep_mask]

        n_low = int(low_harmonics.size)
        if n_low == 0:
            return {
                "low_harmonics": low_harmonics,
                "delta_phi_low_hbkr": np.full((0,) + c.shape[1:], np.nan, dtype=float),
                "Z_b_hkr": self._complex_nan((0, c.shape[2], c.shape[3])),
                "PLV_b_hkr": np.full((0, c.shape[2], c.shape[3]), np.nan, dtype=float),
                "Z_kr_hb": self._complex_nan((0, c.shape[1])),
                "PLV_kr_hb": np.full((0, c.shape[1]), np.nan, dtype=float),
            }

        c_low = c[keep_mask]
        delta_phi = np.angle(c_low)

        Z_b_hkr = self._complex_nan((n_low, c.shape[2], c.shape[3]))
        PLV_b_hkr = np.full((n_low, c.shape[2], c.shape[3]), np.nan, dtype=float)
        Z_kr_hb = self._complex_nan((n_low, c.shape[1]))
        PLV_kr_hb = np.full((n_low, c.shape[1]), np.nan, dtype=float)

        for i in range(n_low):
            Zb = self._complex_resultant(delta_phi[i], axis=0)
            Z_b_hkr[i] = Zb
            valid_b = self._isfinite_complex(Zb)
            PLV_b_hkr[i, valid_b] = np.abs(Zb[valid_b])

            B = delta_phi[i].shape[0]
            for b in range(B):
                Z_kr_hb[i, b] = self._complex_resultant(delta_phi[i, b], axis=(0, 1))
            valid_kr = self._isfinite_complex(Z_kr_hb[i])
            PLV_kr_hb[i, valid_kr] = np.abs(Z_kr_hb[i, valid_kr])

        return {
            "low_harmonics": low_harmonics,
            "delta_phi_low_hbkr": delta_phi,
            "Z_b_hkr": Z_b_hkr,
            "PLV_b_hkr": PLV_b_hkr,
            "Z_kr_hb": Z_kr_hb,
            "PLV_kr_hb": PLV_kr_hb,
        }

    def _spread_metrics(self, c: np.ndarray, means: dict) -> dict:
        valid = self._isfinite_complex(c)
        cbar_b_hkr = means["cbar_b_hkr"]
        cbar_kr_hb = means["cbar_kr_hb"]

        diff_b = np.where(valid, c - cbar_b_hkr[:, None, :, :], np.nan + 1j * np.nan)
        num_b = np.sum(
            np.where(self._isfinite_complex(diff_b), np.abs(diff_b) ** 2, 0.0), axis=1
        )
        den_b = np.sum(np.where(valid, np.abs(c) ** 2, 0.0), axis=1)
        S_b_hkr = np.full(den_b.shape, np.nan, dtype=float)
        ok_b = den_b > 0
        S_b_hkr[ok_b] = np.sqrt(num_b[ok_b] / den_b[ok_b])

        diff_kr = np.where(
            valid, c - cbar_kr_hb[:, :, None, None], np.nan + 1j * np.nan
        )
        num_kr = np.sum(
            np.where(self._isfinite_complex(diff_kr), np.abs(diff_kr) ** 2, 0.0),
            axis=(2, 3),
        )
        den_kr = np.sum(np.where(valid, np.abs(c) ** 2, 0.0), axis=(2, 3))
        S_kr_hb = np.full(den_kr.shape, np.nan, dtype=float)
        ok_kr = den_kr > 0
        S_kr_hb[ok_kr] = np.sqrt(num_kr[ok_kr] / den_kr[ok_kr])

        return {"S_b_hkr": S_b_hkr, "S_kr_hb": S_kr_hb}

    def _occupancy_metrics(self, c: np.ndarray) -> dict:
        mag = np.where(self._isfinite_complex(c), np.abs(c), np.nan)

        A_b_hkr = self._safe_nanmedian(mag, axis=1)  # (H-1,K,R)
        Hm1, K, R = A_b_hkr.shape
        A_b_flat = A_b_hkr.reshape(Hm1, K * R)

        N_kr_h = self._entropy_effective_number(A_b_flat, axis=1)
        N_kr_h_norm = np.full_like(N_kr_h, np.nan, dtype=float)
        if K * R > 0:
            valid = np.isfinite(N_kr_h)
            N_kr_h_norm[valid] = N_kr_h[valid] / float(K * R)

        A_kr_hb = self._safe_nanmedian(
            mag.reshape(mag.shape[0], mag.shape[1], -1), axis=2
        )  # (H-1,B)

        N_b_h = self._entropy_effective_number(A_kr_hb, axis=1)
        N_b_h_norm = np.full_like(N_b_h, np.nan, dtype=float)
        B = c.shape[1]
        if B > 0:
            valid = np.isfinite(N_b_h)
            N_b_h_norm[valid] = N_b_h[valid] / float(B)

        return {
            "A_b_hkr": A_b_hkr,
            "A_kr_hb": A_kr_hb,
            "N_kr_h": N_kr_h,
            "N_kr_h_norm": N_kr_h_norm,
            "N_b_h": N_b_h,
            "N_b_h_norm": N_b_h_norm,
        }

    # ----------------------------
    # Summaries
    # ----------------------------
    def _summary_over_locations(self, x: np.ndarray) -> dict:
        x = np.asarray(x)
        flat = x.reshape(x.shape[0], -1)
        vals = np.abs(flat) if np.iscomplexobj(flat) else flat.astype(float)

        return {
            "median": self._safe_nanmedian(vals, axis=1),
            "std": self._safe_nanstd(vals, axis=1),
        }

    def _summary_over_beats(self, x: np.ndarray) -> dict:
        x = np.asarray(x)
        vals = np.abs(x) if np.iscomplexobj(x) else x.astype(float)
        return {
            "median": self._safe_nanmedian(vals, axis=1),
            "std": self._safe_nanstd(vals, axis=1),
        }

    # ----------------------------
    # Representation-level computation
    # ----------------------------
    def _compute_representation_metrics(
        self, v_block: np.ndarray, T_vec: np.ndarray
    ) -> dict:
        harm = self._harmonic_coefficients_block(v_block, T_vec)
        V = harm["V"]
        H = harm["H"]
        valid_waveform_mask = harm["valid_waveform_mask"]

        norm = self._normalized_harmonics(V)
        c = norm["c"]
        stable_mask = norm["stable_mask"]
        harmonics = norm["harmonics"]

        means = self._beat_and_location_means(c)
        coherence = self._coherence_metrics(c)
        phase = self._low_order_phase_metrics(c, harmonics)
        spread = self._spread_metrics(c, means)
        occupancy = self._occupancy_metrics(c)

        summary = {
            "abs_cbar_b_over_kr": self._summary_over_locations(means["cbar_b_hkr"]),
            "Gamma_b_over_kr": self._summary_over_locations(coherence["Gamma_b_hkr"]),
            "S_b_over_kr": self._summary_over_locations(spread["S_b_hkr"]),
            "abs_cbar_kr_over_b": self._summary_over_beats(means["cbar_kr_hb"]),
            "Gamma_kr_over_b": self._summary_over_beats(coherence["Gamma_kr_hb"]),
            "S_kr_over_b": self._summary_over_beats(spread["S_kr_hb"]),
        }

        if phase["low_harmonics"].size > 0:
            summary["PLV_b_over_kr"] = self._summary_over_locations(phase["PLV_b_hkr"])
            summary["PLV_kr_over_b"] = self._summary_over_beats(phase["PLV_kr_hb"])

        return {
            "H": int(H),
            "harmonics": harmonics,
            "low_harmonics": phase["low_harmonics"],
            "valid_waveform_mask_bkr": valid_waveform_mask.astype(np.uint8),
            "stable_mask_hbkr": stable_mask.astype(np.uint8),
            "V_hbkr": V,
            "V_magnitude_hbkr": np.abs(V),
            "V_phase_hbkr": np.angle(V),
            "c_hbkr": c,
            "c_magnitude_hbkr": np.abs(c),
            "c_phase_hbkr": np.angle(c),
            "cbar_b_hkr": means["cbar_b_hkr"],
            "cbar_b_magnitude_hkr": np.abs(means["cbar_b_hkr"]),
            "cbar_b_phase_hkr": np.angle(means["cbar_b_hkr"]),
            "cbar_kr_hb": means["cbar_kr_hb"],
            "cbar_kr_magnitude_hb": np.abs(means["cbar_kr_hb"]),
            "cbar_kr_phase_hb": np.angle(means["cbar_kr_hb"]),
            "Gamma_b_hkr": coherence["Gamma_b_hkr"],
            "Gamma_kr_hb": coherence["Gamma_kr_hb"],
            "delta_phi_low_hbkr": phase["delta_phi_low_hbkr"],
            "Z_b_hkr": phase["Z_b_hkr"],
            "PLV_b_hkr": phase["PLV_b_hkr"],
            "Z_kr_hb": phase["Z_kr_hb"],
            "PLV_kr_hb": phase["PLV_kr_hb"],
            "S_b_hkr": spread["S_b_hkr"],
            "S_kr_hb": spread["S_kr_hb"],
            "A_b_hkr": occupancy["A_b_hkr"],
            "A_kr_hb": occupancy["A_kr_hb"],
            "N_kr_h": occupancy["N_kr_h"],
            "N_kr_h_norm": occupancy["N_kr_h_norm"],
            "N_b_h": occupancy["N_b_h"],
            "N_b_h_norm": occupancy["N_b_h_norm"],
            "summary": summary,
        }

    # ----------------------------
    # Output storage helpers
    # ----------------------------
    def _store_float(self, metrics: dict, path: str, arr, attrs: dict | None = None):
        val = self._f32(arr)
        metrics[path] = with_attrs(val, attrs or {})

    def _store_int(self, metrics: dict, path: str, arr, attrs: dict | None = None):
        val = self._i32(arr)
        metrics[path] = with_attrs(val, attrs or {})

    def _store_u8(self, metrics: dict, path: str, arr, attrs: dict | None = None):
        val = self._u8(arr)
        metrics[path] = with_attrs(val, attrs or {})

    def _store_complex_split(
        self, metrics: dict, path: str, arr, attrs: dict | None = None
    ):
        arr = np.asarray(arr, dtype=np.complex128)
        metrics[f"{path}_real"] = with_attrs(self._f32(np.real(arr)), attrs or {})
        metrics[f"{path}_imag"] = with_attrs(self._f32(np.imag(arr)), attrs or {})

    # ----------------------------
    # Output packing
    # ----------------------------
    def _pack_representation_outputs(
        self, metrics: dict, prefix: str, rep_name: str, out: dict
    ) -> None:
        base = f"{prefix}/{rep_name}"

        self._store_int(metrics, f"{base}/params/H_MAX", self.H_MAX)
        self._store_float(metrics, f"{base}/params/eps", self.eps)
        self._store_float(
            metrics,
            f"{base}/params/fundamental_abs_threshold",
            self.fundamental_abs_threshold,
        )

        self._store_int(
            metrics,
            f"{base}/axes/harmonics",
            out["harmonics"],
            {
                "definition": [
                    "Higher-harmonic index array corresponding to h=2..H for normalized coefficients."
                ]
            },
        )
        self._store_int(
            metrics,
            f"{base}/axes/low_harmonics",
            out["low_harmonics"],
            {
                "definition": ["Low-order harmonic index array restricted to h in {2,3}."]
            },
        )

        self._store_u8(
            metrics,
            f"{base}/masks/valid_waveform_mask_bkr",
            out["valid_waveform_mask_bkr"],
            {
                "definition": [
                    "1 where the input beat/segment waveform was finite and FFT was computed."
                ]
            },
        )
        self._store_u8(
            metrics,
            f"{base}/masks/stable_mask_hbkr",
            out["stable_mask_hbkr"],
            {
                "definition": [
                    "1 where c_hbkr = V_hbkr / V_1bkr was computed with stable fundamental V_1bkr."
                ]
            },
        )

        self._store_complex_split(
            metrics,
            f"{base}/harmonics/V_hbkr",
            out["V_hbkr"],
            {
                "definition": ["Complex Fourier coefficients V_hbkr for harmonics h=0..H."],
                "layout": ["(harmonic, beat, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/harmonics/V_magnitude_hbkr",
            out["V_magnitude_hbkr"],
            {
                "definition": ["|V_hbkr| for harmonics h=0..H."],
                "layout": ["(harmonic, beat, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/harmonics/V_phase_hbkr",
            out["V_phase_hbkr"],
            {
                "definition": ["arg(V_hbkr) for harmonics h=0..H."],
                "layout": ["(harmonic, beat, branch, radius)"],
            },
        )

        self._store_complex_split(
            metrics,
            f"{base}/normalized/c_hbkr",
            out["c_hbkr"],
            {
                "definition": [r"Normalized higher harmonics c_hbkr = V_hbkr / V_1bkr for h=2..H."],
                "layout": ["(higher_harmonic, beat, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/normalized/c_magnitude_hbkr",
            out["c_magnitude_hbkr"],
            {
                "definition": [r"|c_hbkr| with c_hbkr = V_hbkr / V_1bkr."],
                "layout": ["(higher_harmonic, beat, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/normalized/c_phase_hbkr",
            out["c_phase_hbkr"],
            {
                "definition": [r"arg(c_hbkr), i.e. the phase of harmonic h relative to the fundamental."],
                "layout": ["(higher_harmonic, beat, branch, radius)"],
            },
        )

        self._store_complex_split(
            metrics,
            f"{base}/beat_aggregated/cbar_b_hkr",
            out["cbar_b_hkr"],
            {
                "definition": [r"Beat-aggregated local mean \bar c^(b)_{hkr} = mean_b(c_hbkr)."],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/beat_aggregated/cbar_b_magnitude_hkr",
            out["cbar_b_magnitude_hkr"],
            {
                "definition": [r"| \bar c^(b)_{hkr} |."],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/beat_aggregated/cbar_b_phase_hkr",
            out["cbar_b_phase_hkr"],
            {
                "definition": [r"arg( \bar c^(b)_{hkr} )."],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )

        self._store_complex_split(
            metrics,
            f"{base}/location_aggregated/cbar_kr_hb",
            out["cbar_kr_hb"],
            {
                "definition": [r"Location-aggregated beat mean \bar c^(kr)_{hb} = mean_{k,r}(c_hbkr)."],
                "layout": ["(higher_harmonic, beat)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/location_aggregated/cbar_kr_magnitude_hb",
            out["cbar_kr_magnitude_hb"],
            {
                "definition": [r"| \bar c^(kr)_{hb} |."],
                "layout": ["(higher_harmonic, beat)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/location_aggregated/cbar_kr_phase_hb",
            out["cbar_kr_phase_hb"],
            {
                "definition": [r"arg( \bar c^(kr)_{hb} )."],
                "layout": ["(higher_harmonic, beat)"],
            },
        )

        self._store_float(
            metrics,
            f"{base}/beat_aggregated/Gamma_b_hkr",
            out["Gamma_b_hkr"],
            {
                "definition": [
                    r"Beat coherence \Gamma^(b)_{hkr} = |sum_b c_hbkr| / sum_b |c_hbkr|."
                ],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/location_aggregated/Gamma_kr_hb",
            out["Gamma_kr_hb"],
            {
                "definition": [
                    r"Location coherence \Gamma^(kr)_{hb} = |sum_{k,r} c_hbkr| / sum_{k,r} |c_hbkr|."
                ],
                "layout": ["(higher_harmonic, beat)"],
            },
        )

        self._store_float(
            metrics,
            f"{base}/low_order_phase/delta_phi_low_hbkr",
            out["delta_phi_low_hbkr"],
            {
                "definition": [r"Low-order relative phases \Delta\phi_hbkr = arg(c_hbkr) for h in {2,3}."],
                "layout": ["(low_harmonic, beat, branch, radius)"],
            },
        )
        self._store_complex_split(
            metrics,
            f"{base}/low_order_phase/Z_b_hkr",
            out["Z_b_hkr"],
            {
                "definition": [r"Beat-aggregated circular resultant Z^(b)_{hkr} for h in {2,3}."],
                "layout": ["(low_harmonic, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/low_order_phase/PLV_b_hkr",
            out["PLV_b_hkr"],
            {
                "definition": [r"Beat phase-locking value PLV^(b)_{hkr} = |Z^(b)_{hkr}| for h in {2,3}."],
                "layout": ["(low_harmonic, branch, radius)"],
            },
        )
        self._store_complex_split(
            metrics,
            f"{base}/low_order_phase/Z_kr_hb",
            out["Z_kr_hb"],
            {
                "definition": [r"Location-aggregated circular resultant Z^(kr)_{hb} for h in {2,3}."],
                "layout": ["(low_harmonic, beat)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/low_order_phase/PLV_kr_hb",
            out["PLV_kr_hb"],
            {
                "definition": [r"Location phase-locking value PLV^(kr)_{hb} = |Z^(kr)_{hb}| for h in {2,3}."],
                "layout": ["(low_harmonic, beat)"],
            },
        )

        self._store_float(
            metrics,
            f"{base}/beat_aggregated/S_b_hkr",
            out["S_b_hkr"],
            {
                "definition": [
                    r"Beat-varying spread S^(b)_{hkr} around the beat-aggregated local mean."
                ],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/location_aggregated/S_kr_hb",
            out["S_kr_hb"],
            {
                "definition": [
                    r"Location-varying spread S^(kr)_{hb} around the location-aggregated beat mean."
                ],
                "layout": ["(higher_harmonic, beat)"],
            },
        )

        self._store_float(
            metrics,
            f"{base}/occupancy/A_b_hkr",
            out["A_b_hkr"],
            {
                "definition": [r"A^(b)_{hkr} = median_b |c_hbkr|."],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/occupancy/A_kr_hb",
            out["A_kr_hb"],
            {
                "definition": [r"A^(kr)_{hb} = median_{k,r} |c_hbkr|."],
                "layout": ["(higher_harmonic, beat)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/occupancy/N_kr_h",
            out["N_kr_h"],
            {
                "definition": [r"Effective spatial occupancy N^(kr)_h = exp(-sum_{k,r} p log p)."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/occupancy/N_kr_h_norm",
            out["N_kr_h_norm"],
            {
                "definition": [r"Normalized spatial occupancy N^(kr)_h / N_(KR)."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/occupancy/N_b_h",
            out["N_b_h"],
            {
                "definition": [r"Effective beat occupancy N^(b)_h = exp(-sum_b p log p)."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        self._store_float(
            metrics,
            f"{base}/occupancy/N_b_h_norm",
            out["N_b_h_norm"],
            {
                "definition": [r"Normalized beat occupancy N^(b)_h / N_B."],
                "layout": ["(higher_harmonic,)"],
            },
        )

        summary = out["summary"]

        def store_summary(key: str, desc: str, summ: dict, axis_desc: str):
            self._store_float(
                metrics,
                f"{base}/summary/{key}/median",
                summ["median"],
                {
                    "definition": [f"Median summary of {desc} over {axis_desc}."],
                    "layout": ["(harmonic,)"],
                },
            )
            self._store_float(
                metrics,
                f"{base}/summary/{key}/std",
                summ["std"],
                {
                    "definition": [f"Standard deviation summary of {desc} over {axis_desc}."],
                    "layout": ["(harmonic,)"],
                },
            )

        store_summary(
            "abs_cbar_b_over_kr",
            r"| \bar c^(b)_{hkr} |",
            summary["abs_cbar_b_over_kr"],
            "(branch, radius)",
        )
        store_summary(
            "Gamma_b_over_kr",
            r"\Gamma^(b)_{hkr}",
            summary["Gamma_b_over_kr"],
            "(branch, radius)",
        )
        store_summary(
            "S_b_over_kr",
            r"S^(b)_{hkr}",
            summary["S_b_over_kr"],
            "(branch, radius)",
        )
        store_summary(
            "abs_cbar_kr_over_b",
            r"| \bar c^(kr)_{hb} |",
            summary["abs_cbar_kr_over_b"],
            "beats",
        )
        store_summary(
            "Gamma_kr_over_b",
            r"\Gamma^(kr)_{hb}",
            summary["Gamma_kr_over_b"],
            "beats",
        )
        store_summary(
            "S_kr_over_b",
            r"S^(kr)_{hb}",
            summary["S_kr_over_b"],
            "beats",
        )

        if "PLV_b_over_kr" in summary:
            store_summary(
                "PLV_b_over_kr",
                r"PLV^(b)_{hkr}",
                summary["PLV_b_over_kr"],
                "(branch, radius)",
            )
        if "PLV_kr_over_b" in summary:
            store_summary(
                "PLV_kr_over_b",
                r"PLV^(kr)_{hb}",
                summary["PLV_kr_over_b"],
                "beats",
            )

    # ----------------------------
    # Entry point
    # ----------------------------
    def run(self, h5file) -> ProcessResult:
        if self.T_input not in h5file:
            raise KeyError(f"Missing beat-period input: {self.T_input}")

        T = np.asarray(h5file[self.T_input], dtype=float)
        metrics = {}

        vessel_configs = [
            {
                "prefix": "artery/by_segment",
                "input_path": self.v_raw_segment_input_artery,
            },
            {
                "prefix": "vein/by_segment",
                "input_path": self.v_raw_segment_input_vein,
            },
        ]

        for cfg in vessel_configs:
            if cfg["input_path"] not in h5file:
                continue

            v_seg = np.asarray(h5file[cfg["input_path"]], dtype=float)
            if v_seg.ndim != 4:
                raise ValueError(
                    f"Expected segment block shape (n_t,n_beats,n_branches,n_radii), "
                    f"got {v_seg.shape} for {cfg['input_path']}"
                )

            n_t, n_beats, n_branches, n_radii = v_seg.shape
            T_vec = self._ensure_beat_periods(T, n_beats)

            out = self._compute_representation_metrics(v_seg, T_vec)
            self._pack_representation_outputs(metrics, cfg["prefix"], "raw", out)

            self._store_int(metrics, f"{cfg['prefix']}/params/n_t", n_t)
            self._store_int(metrics, f"{cfg['prefix']}/params/n_beats", n_beats)
            self._store_int(metrics, f"{cfg['prefix']}/params/n_branches", n_branches)
            self._store_int(metrics, f"{cfg['prefix']}/params/n_radii", n_radii)

        return ProcessResult(metrics=metrics)
