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
    higher_harmonic_rel_threshold = 5e-2

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

    @staticmethod
    def _count_true(mask: np.ndarray, axis=None) -> np.ndarray:
        return np.sum(np.asarray(mask, dtype=np.uint8), axis=axis, dtype=np.int32)

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
        - V_hbkr: complex array (H+1, B, K, R), harmonics 0..H
        - valid_waveform_bkr_mask: bool array (B, K, R)
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

        V_hbkr = self._complex_nan((H + 1, n_beats, n_branches, n_radii))
        valid_waveform_bkr_mask = np.zeros((n_beats, n_branches, n_radii), dtype=bool)

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

                    V_hbkr[:, b, k, r] = Vf[: H + 1]
                    valid_waveform_bkr_mask[b, k, r] = True

        return {
            "V_hbkr": V_hbkr,
            "H": H,
            "valid_waveform_bkr_mask": valid_waveform_bkr_mask,
        }

    def _normalized_harmonics(self, V_hbkr: np.ndarray) -> dict:
        """
        Compute c_hbkr = V_hbkr / V_1bkr for h=2..H.

        Validity rule:
          |V_h| / |V_1| > higher_harmonic_rel_threshold

        For numerical safety only, the denominator is guarded by eps.
        """
        H = int(V_hbkr.shape[0] - 1)
        n_beats, n_branches, n_radii = V_hbkr.shape[1:]

        if H < 2:
            c_hbkr = self._complex_nan((0, n_beats, n_branches, n_radii))
            valid_c_hbkr_mask = np.zeros((0, n_beats, n_branches, n_radii), dtype=bool)
            harmonics = np.asarray([], dtype=int)
            rel_amp_hbkr = np.full((0, n_beats, n_branches, n_radii), np.nan, dtype=float)
            return {
                "c_hbkr": c_hbkr,
                "valid_c_hbkr_mask": valid_c_hbkr_mask,
                "harmonics": harmonics,
                "rel_amp_hbkr": rel_amp_hbkr,
            }

        V1_bkr = V_hbkr[1]
        valid_den_bkr = self._isfinite_complex(V1_bkr) & (np.abs(V1_bkr) > self.eps)

        c_hbkr = self._complex_nan((H - 1, n_beats, n_branches, n_radii))
        valid_c_hbkr_mask = np.zeros((H - 1, n_beats, n_branches, n_radii), dtype=bool)
        rel_amp_hbkr = np.full((H - 1, n_beats, n_branches, n_radii), np.nan, dtype=float)

        for hi, h in enumerate(range(2, H + 1)):
            Vh_bkr = V_hbkr[h]
            finite_h_bkr = self._isfinite_complex(Vh_bkr) & self._isfinite_complex(V1_bkr)

            rel = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
            rel[finite_h_bkr] = (
                np.abs(Vh_bkr[finite_h_bkr]) / (np.abs(V1_bkr[finite_h_bkr]) + self.eps)
            )
            rel_amp_hbkr[hi] = rel

            ok = finite_h_bkr & valid_den_bkr & (rel > float(self.higher_harmonic_rel_threshold))
            valid_c_hbkr_mask[hi] = ok

            c_h = self._complex_nan((n_beats, n_branches, n_radii))
            c_h[ok] = Vh_bkr[ok] / V1_bkr[ok]
            c_hbkr[hi] = c_h

        harmonics = np.arange(2, H + 1, dtype=int)
        return {
            "c_hbkr": c_hbkr,
            "valid_c_hbkr_mask": valid_c_hbkr_mask,
            "harmonics": harmonics,
            "rel_amp_hbkr": rel_amp_hbkr,
        }

    # ----------------------------
    # Direct metrics
    # ----------------------------
    def _beat_and_location_means(self, c_hbkr: np.ndarray, valid_c_hbkr_mask: np.ndarray) -> dict:
        """
        c_hbkr shape: (H-1, B, K, R)
        """
        cbar_b_over_hkr = self._complex_mean(c_hbkr, axis=1)      # (H-1, K, R)
        cbar_kr_over_hb = self._complex_mean(c_hbkr, axis=(2, 3)) # (H-1, B)

        valid_b_count_over_hkr = self._count_true(valid_c_hbkr_mask, axis=1)
        valid_kr_count_over_hb = self._count_true(valid_c_hbkr_mask, axis=(2, 3))

        return {
            "cbar_b_over_hkr": cbar_b_over_hkr,
            "cbar_kr_over_hb": cbar_kr_over_hb,
            "valid_b_count_over_hkr": valid_b_count_over_hkr,
            "valid_kr_count_over_hb": valid_kr_count_over_hb,
        }

    def _coherence_metrics(self, c_hbkr: np.ndarray) -> dict:
        valid = self._isfinite_complex(c_hbkr)
        abs_c = np.where(valid, np.abs(c_hbkr), 0.0)

        sum_b_over_hkr = np.sum(np.where(valid, c_hbkr, 0.0 + 0.0j), axis=1)
        den_b_over_hkr = np.sum(abs_c, axis=1)
        Gamma_b_over_hkr = np.full(den_b_over_hkr.shape, np.nan, dtype=float)
        ok_b = den_b_over_hkr > 0
        Gamma_b_over_hkr[ok_b] = np.abs(sum_b_over_hkr[ok_b]) / den_b_over_hkr[ok_b]

        sum_kr_over_hb = np.sum(np.where(valid, c_hbkr, 0.0 + 0.0j), axis=(2, 3))
        den_kr_over_hb = np.sum(abs_c, axis=(2, 3))
        Gamma_kr_over_hb = np.full(den_kr_over_hb.shape, np.nan, dtype=float)
        ok_kr = den_kr_over_hb > 0
        Gamma_kr_over_hb[ok_kr] = np.abs(sum_kr_over_hb[ok_kr]) / den_kr_over_hb[ok_kr]

        return {
            "Gamma_b_over_hkr": Gamma_b_over_hkr,
            "Gamma_kr_over_hb": Gamma_kr_over_hb,
        }

    def _low_order_phase_metrics(
        self, c_hbkr: np.ndarray, valid_c_hbkr_mask: np.ndarray, harmonics: np.ndarray
    ) -> dict:
        keep_mask = np.isin(harmonics, np.asarray([2, 3], dtype=int))
        low_harmonics = harmonics[keep_mask]

        n_low = int(low_harmonics.size)
        if n_low == 0:
            return {
                "low_harmonics": low_harmonics,
                "delta_phi_low_hbkr": np.full((0,) + c_hbkr.shape[1:], np.nan, dtype=float),
                "Z_b_over_hkr": self._complex_nan((0, c_hbkr.shape[2], c_hbkr.shape[3])),
                "PLV_b_over_hkr": np.full((0, c_hbkr.shape[2], c_hbkr.shape[3]), np.nan, dtype=float),
                "Z_kr_over_hb": self._complex_nan((0, c_hbkr.shape[1])),
                "PLV_kr_over_hb": np.full((0, c_hbkr.shape[1]), np.nan, dtype=float),
                "valid_b_count_over_hkr": np.zeros((0, c_hbkr.shape[2], c_hbkr.shape[3]), dtype=np.int32),
                "valid_kr_count_over_hb": np.zeros((0, c_hbkr.shape[1]), dtype=np.int32),
            }

        c_low_hbkr = c_hbkr[keep_mask]
        valid_low_hbkr_mask = valid_c_hbkr_mask[keep_mask]
        delta_phi_low_hbkr = np.angle(c_low_hbkr)

        Z_b_over_hkr = self._complex_nan((n_low, c_hbkr.shape[2], c_hbkr.shape[3]))
        PLV_b_over_hkr = np.full((n_low, c_hbkr.shape[2], c_hbkr.shape[3]), np.nan, dtype=float)
        Z_kr_over_hb = self._complex_nan((n_low, c_hbkr.shape[1]))
        PLV_kr_over_hb = np.full((n_low, c_hbkr.shape[1]), np.nan, dtype=float)

        valid_b_count_over_hkr = self._count_true(valid_low_hbkr_mask, axis=1)
        valid_kr_count_over_hb = self._count_true(valid_low_hbkr_mask, axis=(2, 3))

        for i in range(n_low):
            Zb_over_kr = self._complex_resultant(delta_phi_low_hbkr[i], axis=0)
            Z_b_over_hkr[i] = Zb_over_kr
            valid_b = self._isfinite_complex(Zb_over_kr)
            PLV_b_over_hkr[i, valid_b] = np.abs(Zb_over_kr[valid_b])

            n_beats = delta_phi_low_hbkr[i].shape[0]
            for b in range(n_beats):
                Z_kr_over_hb[i, b] = self._complex_resultant(
                    delta_phi_low_hbkr[i, b], axis=(0, 1)
                )
            valid_kr = self._isfinite_complex(Z_kr_over_hb[i])
            PLV_kr_over_hb[i, valid_kr] = np.abs(Z_kr_over_hb[i, valid_kr])

        return {
            "low_harmonics": low_harmonics,
            "delta_phi_low_hbkr": delta_phi_low_hbkr,
            "Z_b_over_hkr": Z_b_over_hkr,
            "PLV_b_over_hkr": PLV_b_over_hkr,
            "Z_kr_over_hb": Z_kr_over_hb,
            "PLV_kr_over_hb": PLV_kr_over_hb,
            "valid_b_count_over_hkr": valid_b_count_over_hkr,
            "valid_kr_count_over_hb": valid_kr_count_over_hb,
        }

    def _spread_metrics(self, c_hbkr: np.ndarray, means: dict) -> dict:
        valid = self._isfinite_complex(c_hbkr)
        cbar_b_over_hkr = means["cbar_b_over_hkr"]
        cbar_kr_over_hb = means["cbar_kr_over_hb"]

        diff_b = np.where(
            valid, c_hbkr - cbar_b_over_hkr[:, None, :, :], np.nan + 1j * np.nan
        )
        num_b = np.sum(
            np.where(self._isfinite_complex(diff_b), np.abs(diff_b) ** 2, 0.0), axis=1
        )
        den_b = np.sum(np.where(valid, np.abs(c_hbkr) ** 2, 0.0), axis=1)
        S_b_over_hkr = np.full(den_b.shape, np.nan, dtype=float)
        ok_b = den_b > 0
        S_b_over_hkr[ok_b] = np.sqrt(num_b[ok_b] / den_b[ok_b])

        diff_kr = np.where(
            valid, c_hbkr - cbar_kr_over_hb[:, :, None, None], np.nan + 1j * np.nan
        )
        num_kr = np.sum(
            np.where(self._isfinite_complex(diff_kr), np.abs(diff_kr) ** 2, 0.0),
            axis=(2, 3),
        )
        den_kr = np.sum(np.where(valid, np.abs(c_hbkr) ** 2, 0.0), axis=(2, 3))
        S_kr_over_hb = np.full(den_kr.shape, np.nan, dtype=float)
        ok_kr = den_kr > 0
        S_kr_over_hb[ok_kr] = np.sqrt(num_kr[ok_kr] / den_kr[ok_kr])

        return {
            "S_b_over_hkr": S_b_over_hkr,
            "S_kr_over_hb": S_kr_over_hb,
        }

    def _occupancy_metrics(self, c_hbkr: np.ndarray) -> dict:
        abs_c_hbkr = np.where(self._isfinite_complex(c_hbkr), np.abs(c_hbkr), np.nan)

        A_b_over_hkr = self._safe_nanmedian(abs_c_hbkr, axis=1)  # (H-1,K,R)
        valid_A_b_over_hkr_mask = np.isfinite(A_b_over_hkr)
        Hm1, n_branches, n_radii = A_b_over_hkr.shape
        A_b_over_hkr_flat = A_b_over_hkr.reshape(Hm1, n_branches * n_radii)

        N_kr_over_h = self._entropy_effective_number(A_b_over_hkr_flat, axis=1)
        valid_kr_count_over_h = self._count_true(
            valid_A_b_over_hkr_mask.reshape(Hm1, -1), axis=1
        )
        N_kr_over_h_norm = np.full_like(N_kr_over_h, np.nan, dtype=float)
        ok_kr = valid_kr_count_over_h > 0
        N_kr_over_h_norm[ok_kr] = (
            N_kr_over_h[ok_kr] / valid_kr_count_over_h[ok_kr].astype(float)
        )

        A_kr_over_hb = self._safe_nanmedian(
            abs_c_hbkr.reshape(abs_c_hbkr.shape[0], abs_c_hbkr.shape[1], -1), axis=2
        )  # (H-1,B)
        valid_A_kr_over_hb_mask = np.isfinite(A_kr_over_hb)

        N_b_over_h = self._entropy_effective_number(A_kr_over_hb, axis=1)
        valid_b_count_over_h = self._count_true(valid_A_kr_over_hb_mask, axis=1)
        N_b_over_h_norm = np.full_like(N_b_over_h, np.nan, dtype=float)
        ok_b = valid_b_count_over_h > 0
        N_b_over_h_norm[ok_b] = (
            N_b_over_h[ok_b] / valid_b_count_over_h[ok_b].astype(float)
        )

        return {
            "A_b_over_hkr": A_b_over_hkr,
            "A_kr_over_hb": A_kr_over_hb,
            "valid_A_b_over_hkr_mask": valid_A_b_over_hkr_mask,
            "valid_A_kr_over_hb_mask": valid_A_kr_over_hb_mask,
            "N_kr_over_h": N_kr_over_h,
            "N_kr_over_h_norm": N_kr_over_h_norm,
            "N_b_over_h": N_b_over_h,
            "N_b_over_h_norm": N_b_over_h_norm,
            "valid_kr_count_over_h": valid_kr_count_over_h,
            "valid_b_count_over_h": valid_b_count_over_h,
        }

    # ----------------------------
    # Summaries
    # ----------------------------
    def _summary_over_locations(self, x: np.ndarray) -> dict:
        """
        Summarize a descriptor indexed by (h, k, r) over (k, r), returning (h,).
        Also report the valid location count used for the median/std reduction.
        """
        x = np.asarray(x)
        flat = x.reshape(x.shape[0], -1)

        if np.iscomplexobj(flat):
            valid = self._isfinite_complex(flat)
            vals = np.abs(flat)
        else:
            vals = flat.astype(float)
            valid = np.isfinite(vals)

        vals = np.where(valid, vals, np.nan)

        return {
            "median": self._safe_nanmedian(vals, axis=1),
            "std": self._safe_nanstd(vals, axis=1),
            "valid_kr_count_over_h": self._count_true(valid, axis=1),
        }

    def _summary_over_beats(self, x: np.ndarray) -> dict:
        """
        Summarize a descriptor indexed by (h, b) over b, returning (h,).
        Also report the valid beat count used for the median/std reduction.
        """
        x = np.asarray(x)

        if np.iscomplexobj(x):
            valid = self._isfinite_complex(x)
            vals = np.abs(x)
        else:
            vals = x.astype(float)
            valid = np.isfinite(vals)

        vals = np.where(valid, vals, np.nan)

        return {
            "median": self._safe_nanmedian(vals, axis=1),
            "std": self._safe_nanstd(vals, axis=1),
            "valid_b_count_over_h": self._count_true(valid, axis=1),
        }

    # ----------------------------
    # Main block computation
    # ----------------------------
    def _compute_representation_metrics(
        self, v_block: np.ndarray, T_vec: np.ndarray
    ) -> dict:
        harm = self._harmonic_coefficients_block(v_block, T_vec)
        V_hbkr = harm["V_hbkr"]
        H = harm["H"]
        valid_waveform_bkr_mask = harm["valid_waveform_bkr_mask"]

        norm = self._normalized_harmonics(V_hbkr)
        c_hbkr = norm["c_hbkr"]
        valid_c_hbkr_mask = norm["valid_c_hbkr_mask"]
        harmonics = norm["harmonics"]
        rel_amp_hbkr = norm["rel_amp_hbkr"]

        means = self._beat_and_location_means(c_hbkr, valid_c_hbkr_mask)
        coherence = self._coherence_metrics(c_hbkr)
        phase = self._low_order_phase_metrics(c_hbkr, valid_c_hbkr_mask, harmonics)
        spread = self._spread_metrics(c_hbkr, means)
        occupancy = self._occupancy_metrics(c_hbkr)

        summary = {
            "abs_cbar_b_over_kr": self._summary_over_locations(
                means["cbar_b_over_hkr"]
            ),
            "Gamma_b_over_kr": self._summary_over_locations(
                coherence["Gamma_b_over_hkr"]
            ),
            "S_b_over_kr": self._summary_over_locations(spread["S_b_over_hkr"]),
            "abs_cbar_kr_over_b": self._summary_over_beats(
                means["cbar_kr_over_hb"]
            ),
            "Gamma_kr_over_b": self._summary_over_beats(
                coherence["Gamma_kr_over_hb"]
            ),
            "S_kr_over_b": self._summary_over_beats(spread["S_kr_over_hb"]),
        }

        if phase["low_harmonics"].size > 0:
            summary["PLV_b_over_kr"] = self._summary_over_locations(
                phase["PLV_b_over_hkr"]
            )
            summary["PLV_kr_over_b"] = self._summary_over_beats(
                phase["PLV_kr_over_hb"]
            )

        return {
            "H": int(H),
            "harmonics": harmonics,
            "low_harmonics": phase["low_harmonics"],
            "valid_waveform_bkr_mask": valid_waveform_bkr_mask.astype(np.uint8),
            "valid_c_hbkr_mask": valid_c_hbkr_mask.astype(np.uint8),
            "V_hbkr": V_hbkr,
            "c_hbkr": c_hbkr,
            "rel_amp_hbkr": rel_amp_hbkr,
            "cbar_b_over_hkr": means["cbar_b_over_hkr"],
            "cbar_kr_over_hb": means["cbar_kr_over_hb"],
            "valid_b_count_over_hkr": means["valid_b_count_over_hkr"],
            "valid_kr_count_over_hb": means["valid_kr_count_over_hb"],
            "Gamma_b_over_hkr": coherence["Gamma_b_over_hkr"],
            "Gamma_kr_over_hb": coherence["Gamma_kr_over_hb"],
            "delta_phi_low_hbkr": phase["delta_phi_low_hbkr"],
            "Z_b_over_hkr": phase["Z_b_over_hkr"],
            "PLV_b_over_hkr": phase["PLV_b_over_hkr"],
            "Z_kr_over_hb": phase["Z_kr_over_hb"],
            "PLV_kr_over_hb": phase["PLV_kr_over_hb"],
            "phase_valid_b_count_over_hkr": phase["valid_b_count_over_hkr"],
            "phase_valid_kr_count_over_hb": phase["valid_kr_count_over_hb"],
            "S_b_over_hkr": spread["S_b_over_hkr"],
            "S_kr_over_hb": spread["S_kr_over_hb"],
            "A_b_over_hkr": occupancy["A_b_over_hkr"],
            "A_kr_over_hb": occupancy["A_kr_over_hb"],
            "valid_A_b_over_hkr_mask": occupancy["valid_A_b_over_hkr_mask"].astype(np.uint8),
            "valid_A_kr_over_hb_mask": occupancy["valid_A_kr_over_hb_mask"].astype(np.uint8),
            "N_kr_over_h": occupancy["N_kr_over_h"],
            "N_kr_over_h_norm": occupancy["N_kr_over_h_norm"],
            "N_b_over_h": occupancy["N_b_over_h"],
            "N_b_over_h_norm": occupancy["N_b_over_h_norm"],
            "valid_kr_count_over_h": occupancy["valid_kr_count_over_h"],
            "valid_b_count_over_h": occupancy["valid_b_count_over_h"],
            "summary": summary,
        }

    # ----------------------------
    # Writing helpers
    # ----------------------------
    def _pack_split_complex(
        self,
        metrics: dict,
        path: str,
        z: np.ndarray,
        attrs_common: dict,
    ) -> None:
        metrics[f"{path}_real"] = with_attrs(
            np.asarray(np.real(z), dtype=np.float32), attrs_common
        )
        metrics[f"{path}_imag"] = with_attrs(
            np.asarray(np.imag(z), dtype=np.float32), attrs_common
        )

    def _pack_representation_outputs(
        self, metrics: dict, vessel_prefix: str, out: dict
    ) -> None:
        """
        Write outputs under:
          {vessel_prefix}/by_segment/raw/...
        """
        base = f"{vessel_prefix}/by_segment/raw"

        metrics[f"{base}/params/H_MAX"] = np.asarray(self.H_MAX, dtype=np.int32)
        metrics[f"{base}/params/eps"] = np.asarray(self.eps, dtype=np.float32)
        metrics[f"{base}/params/higher_harmonic_rel_threshold"] = np.asarray(
            self.higher_harmonic_rel_threshold, dtype=np.float32
        )

        metrics[f"{base}/axes/harmonics"] = with_attrs(
            np.asarray(out["harmonics"], dtype=np.int32),
            {
                "definition": [
                    "Higher-harmonic index array corresponding to h=2..H for normalized coefficients."
                ]
            },
        )
        metrics[f"{base}/axes/low_harmonics"] = with_attrs(
            np.asarray(out["low_harmonics"], dtype=np.int32),
            {
                "definition": ["Low-order harmonic index array restricted to h in {2,3}."]
            },
        )

        metrics[f"{base}/masks/valid_waveform_bkr_mask"] = with_attrs(
            np.asarray(out["valid_waveform_bkr_mask"], dtype=np.uint8),
            {
                "definition": [
                    "1 where the input beat/segment waveform was finite and FFT was computed."
                ]
            },
        )
        metrics[f"{base}/masks/valid_c_hbkr_mask"] = with_attrs(
            np.asarray(out["valid_c_hbkr_mask"], dtype=np.uint8),
            {
                "definition": [
                    "1 where c_hbkr = V_hbkr / V_1bkr satisfied the higher-harmonic relative validity criterion."
                ]
            },
        )
        metrics[f"{base}/masks/valid_A_b_over_hkr_mask"] = with_attrs(
            np.asarray(out["valid_A_b_over_hkr_mask"], dtype=np.uint8),
            {
                "definition": [
                    "1 where A_b_over_hkr = median_b |c_hbkr| is finite."
                ]
            },
        )
        metrics[f"{base}/masks/valid_A_kr_over_hb_mask"] = with_attrs(
            np.asarray(out["valid_A_kr_over_hb_mask"], dtype=np.uint8),
            {
                "definition": [
                    "1 where A_kr_over_hb = median_{k,r} |c_hbkr| is finite."
                ]
            },
        )

        self._pack_split_complex(
            metrics,
            f"{base}/harmonics/V_hbkr",
            out["V_hbkr"],
            {
                "definition": ["Complex Fourier coefficients V_hbkr for harmonics h=0..H."],
                "layout": ["(harmonic, beat, branch, radius)"],
            },
        )

        self._pack_split_complex(
            metrics,
            f"{base}/normalized/c_hbkr",
            out["c_hbkr"],
            {
                "definition": [r"Normalized higher harmonics c_hbkr = V_hbkr / V_1bkr for h=2..H."],
                "layout": ["(higher_harmonic, beat, branch, radius)"],
            },
        )
        metrics[f"{base}/normalized/rel_amp_hbkr"] = with_attrs(
            np.asarray(out["rel_amp_hbkr"], dtype=np.float32),
            {
                "definition": [r"Relative higher-harmonic amplitude |V_hbkr| / |V_1bkr| used in the validity rule."],
                "layout": ["(higher_harmonic, beat, branch, radius)"],
            },
        )

        self._pack_split_complex(
            metrics,
            f"{base}/beat_aggregated/cbar_b_over_hkr",
            out["cbar_b_over_hkr"],
            {
                "definition": [r"Beat-aggregated local mean \bar c^{(b)}_{hkr} = mean_b(c_hbkr)."],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        metrics[f"{base}/beat_aggregated/Gamma_b_over_hkr"] = with_attrs(
            np.asarray(out["Gamma_b_over_hkr"], dtype=np.float32),
            {
                "definition": [
                    r"Beat coherence \Gamma^{(b)}_{hkr} = |sum_b c_hbkr| / sum_b |c_hbkr|."
                ],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        metrics[f"{base}/beat_aggregated/S_b_over_hkr"] = with_attrs(
            np.asarray(out["S_b_over_hkr"], dtype=np.float32),
            {
                "definition": [
                    r"Beat-varying spread S^{(b)}_{hkr} around the beat-aggregated local mean."
                ],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        metrics[f"{base}/beat_aggregated/valid_b_count_over_hkr"] = with_attrs(
            np.asarray(out["valid_b_count_over_hkr"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid beats contributing at each (h,k,r) for beat-aggregated reductions."
                ],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )

        self._pack_split_complex(
            metrics,
            f"{base}/location_aggregated/cbar_kr_over_hb",
            out["cbar_kr_over_hb"],
            {
                "definition": [r"Location-aggregated beat mean \bar c^{(kr)}_{hb} = mean_{k,r}(c_hbkr)."],
                "layout": ["(higher_harmonic, beat)"],
            },
        )
        metrics[f"{base}/location_aggregated/Gamma_kr_over_hb"] = with_attrs(
            np.asarray(out["Gamma_kr_over_hb"], dtype=np.float32),
            {
                "definition": [
                    r"Location coherence \Gamma^{(kr)}_{hb} = |sum_{k,r} c_hbkr| / sum_{k,r} |c_hbkr|."
                ],
                "layout": ["(higher_harmonic, beat)"],
            },
        )
        metrics[f"{base}/location_aggregated/S_kr_over_hb"] = with_attrs(
            np.asarray(out["S_kr_over_hb"], dtype=np.float32),
            {
                "definition": [
                    r"Location-varying spread S^{(kr)}_{hb} around the location-aggregated beat mean."
                ],
                "layout": ["(higher_harmonic, beat)"],
            },
        )
        metrics[f"{base}/location_aggregated/valid_kr_count_over_hb"] = with_attrs(
            np.asarray(out["valid_kr_count_over_hb"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid locations contributing at each (h,b) for location-aggregated reductions."
                ],
                "layout": ["(higher_harmonic, beat)"],
            },
        )

        metrics[f"{base}/low_order_phase/delta_phi_low_hbkr"] = with_attrs(
            np.asarray(out["delta_phi_low_hbkr"], dtype=np.float32),
            {
                "definition": [r"Low-order relative phases \Delta\phi_{hbkr} = arg(c_hbkr) for h in {2,3}."],
                "layout": ["(low_harmonic, beat, branch, radius)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/low_order_phase/Z_b_over_hkr",
            out["Z_b_over_hkr"],
            {
                "definition": [r"Beat-aggregated circular resultant Z^{(b)}_{hkr} for h in {2,3}."],
                "layout": ["(low_harmonic, branch, radius)"],
            },
        )
        metrics[f"{base}/low_order_phase/PLV_b_over_hkr"] = with_attrs(
            np.asarray(out["PLV_b_over_hkr"], dtype=np.float32),
            {
                "definition": [r"Beat phase-locking value PLV^{(b)}_{hkr} = |Z^{(b)}_{hkr}| for h in {2,3}."],
                "layout": ["(low_harmonic, branch, radius)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/low_order_phase/Z_kr_over_hb",
            out["Z_kr_over_hb"],
            {
                "definition": [r"Location-aggregated circular resultant Z^{(kr)}_{hb} for h in {2,3}."],
                "layout": ["(low_harmonic, beat)"],
            },
        )
        metrics[f"{base}/low_order_phase/PLV_kr_over_hb"] = with_attrs(
            np.asarray(out["PLV_kr_over_hb"], dtype=np.float32),
            {
                "definition": [r"Location phase-locking value PLV^{(kr)}_{hb} = |Z^{(kr)}_{hb}| for h in {2,3}."],
                "layout": ["(low_harmonic, beat)"],
            },
        )
        metrics[f"{base}/low_order_phase/valid_b_count_over_hkr"] = with_attrs(
            np.asarray(out["phase_valid_b_count_over_hkr"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid beats contributing to low-order phase beat-aggregated reductions."
                ],
                "layout": ["(low_harmonic, branch, radius)"],
            },
        )
        metrics[f"{base}/low_order_phase/valid_kr_count_over_hb"] = with_attrs(
            np.asarray(out["phase_valid_kr_count_over_hb"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid locations contributing to low-order phase location-aggregated reductions."
                ],
                "layout": ["(low_harmonic, beat)"],
            },
        )

        metrics[f"{base}/occupancy/A_b_over_hkr"] = with_attrs(
            np.asarray(out["A_b_over_hkr"], dtype=np.float32),
            {
                "definition": [r"A^{(b)}_{hkr} = median_b |c_hbkr|."],
                "layout": ["(higher_harmonic, branch, radius)"],
            },
        )
        metrics[f"{base}/occupancy/A_kr_over_hb"] = with_attrs(
            np.asarray(out["A_kr_over_hb"], dtype=np.float32),
            {
                "definition": [r"A^{(kr)}_{hb} = median_{k,r} |c_hbkr|."],
                "layout": ["(higher_harmonic, beat)"],
            },
        )
        metrics[f"{base}/occupancy/N_kr_over_h"] = with_attrs(
            np.asarray(out["N_kr_over_h"], dtype=np.float32),
            {
                "definition": [r"Effective spatial occupancy N^{(kr)}_h = exp(-sum_{k,r} p log p)."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        metrics[f"{base}/occupancy/N_kr_over_h_norm"] = with_attrs(
            np.asarray(out["N_kr_over_h_norm"], dtype=np.float32),
            {
                "definition": [r"Normalized spatial occupancy N^{(kr)}_h divided by the number of valid locations for harmonic h."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        metrics[f"{base}/occupancy/N_b_over_h"] = with_attrs(
            np.asarray(out["N_b_over_h"], dtype=np.float32),
            {
                "definition": [r"Effective beat occupancy N^{(b)}_h = exp(-sum_b p log p)."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        metrics[f"{base}/occupancy/N_b_over_h_norm"] = with_attrs(
            np.asarray(out["N_b_over_h_norm"], dtype=np.float32),
            {
                "definition": [r"Normalized beat occupancy N^{(b)}_h divided by the number of valid beats for harmonic h."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        metrics[f"{base}/occupancy/valid_kr_count_over_h"] = with_attrs(
            np.asarray(out["valid_kr_count_over_h"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid locations used to normalize N_kr_over_h_norm for each harmonic h."
                ],
                "layout": ["(higher_harmonic,)"],
            },
        )
        metrics[f"{base}/occupancy/valid_b_count_over_h"] = with_attrs(
            np.asarray(out["valid_b_count_over_h"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid beats used to normalize N_b_over_h_norm for each harmonic h."
                ],
                "layout": ["(higher_harmonic,)"],
            },
        )

        summary = out["summary"]

        def store_summary_locations(key: str, desc: str, summ: dict):
            metrics[f"{base}/summary/{key}/median"] = with_attrs(
                np.asarray(summ["median"], dtype=np.float32),
                {
                    "definition": [f"Median summary of {desc} over (branch, radius)."],
                    "layout": ["(harmonic,)"],
                },
            )
            metrics[f"{base}/summary/{key}/std"] = with_attrs(
                np.asarray(summ["std"], dtype=np.float32),
                {
                    "definition": [f"Standard deviation summary of {desc} over (branch, radius)."],
                    "layout": ["(harmonic,)"],
                },
            )
            metrics[f"{base}/summary/{key}/valid_kr_count_over_h"] = with_attrs(
                np.asarray(summ["valid_kr_count_over_h"], dtype=np.int32),
                {
                    "definition": [
                        f"Number of valid locations used for the median/std summary of {desc} for each harmonic."
                    ],
                    "layout": ["(harmonic,)"],
                },
            )

        def store_summary_beats(key: str, desc: str, summ: dict):
            metrics[f"{base}/summary/{key}/median"] = with_attrs(
                np.asarray(summ["median"], dtype=np.float32),
                {
                    "definition": [f"Median summary of {desc} over beats."],
                    "layout": ["(harmonic,)"],
                },
            )
            metrics[f"{base}/summary/{key}/std"] = with_attrs(
                np.asarray(summ["std"], dtype=np.float32),
                {
                    "definition": [f"Standard deviation summary of {desc} over beats."],
                    "layout": ["(harmonic,)"],
                },
            )
            metrics[f"{base}/summary/{key}/valid_b_count_over_h"] = with_attrs(
                np.asarray(summ["valid_b_count_over_h"], dtype=np.int32),
                {
                    "definition": [
                        f"Number of valid beats used for the median/std summary of {desc} for each harmonic."
                    ],
                    "layout": ["(harmonic,)"],
                },
            )

        store_summary_locations(
            "abs_cbar_b_over_kr",
            r"| \bar c^{(b)}_{hkr} |",
            summary["abs_cbar_b_over_kr"],
        )
        store_summary_locations(
            "Gamma_b_over_kr",
            r"\Gamma^{(b)}_{hkr}",
            summary["Gamma_b_over_kr"],
        )
        store_summary_locations(
            "S_b_over_kr",
            r"S^{(b)}_{hkr}",
            summary["S_b_over_kr"],
        )

        store_summary_beats(
            "abs_cbar_kr_over_b",
            r"| \bar c^{(kr)}_{hb} |",
            summary["abs_cbar_kr_over_b"],
        )
        store_summary_beats(
            "Gamma_kr_over_b",
            r"\Gamma^{(kr)}_{hb}",
            summary["Gamma_kr_over_b"],
        )
        store_summary_beats(
            "S_kr_over_b",
            r"S^{(kr)}_{hb}",
            summary["S_kr_over_b"],
        )

        if "PLV_b_over_kr" in summary:
            store_summary_locations(
                "PLV_b_over_kr",
                r"PLV^{(b)}_{hkr}",
                summary["PLV_b_over_kr"],
            )
        if "PLV_kr_over_b" in summary:
            store_summary_beats(
                "PLV_kr_over_b",
                r"PLV^{(kr)}_{hb}",
                summary["PLV_kr_over_b"],
            )

    def _pack_vessel_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        v_seg: np.ndarray,
        T: np.ndarray,
    ) -> None:
        if v_seg.ndim != 4:
            raise ValueError(
                f"Expected (n_t,n_beats,n_branches,n_radii), got {v_seg.shape}"
            )

        n_t, n_beats, n_branches, n_radii = v_seg.shape
        T_vec = self._ensure_beat_periods(T, n_beats)

        out = self._compute_representation_metrics(v_seg, T_vec)
        self._pack_representation_outputs(metrics, vessel_prefix, out)

        metrics[f"{vessel_prefix}/by_segment/params/n_t"] = np.asarray(
            n_t, dtype=np.int32
        )
        metrics[f"{vessel_prefix}/by_segment/params/n_beats"] = np.asarray(
            n_beats, dtype=np.int32
        )
        metrics[f"{vessel_prefix}/by_segment/params/n_branches"] = np.asarray(
            n_branches, dtype=np.int32
        )
        metrics[f"{vessel_prefix}/by_segment/params/n_radii"] = np.asarray(
            n_radii, dtype=np.int32
        )

    def run(self, h5file) -> ProcessResult:
        T = np.asarray(h5file[self.T_input])
        metrics = {}

        vessel_configs = [
            {
                "prefix": "artery",
                "v_raw_segment_input": self.v_raw_segment_input_artery,
            },
            {
                "prefix": "vein",
                "v_raw_segment_input": self.v_raw_segment_input_vein,
            },
        ]

        for cfg in vessel_configs:
            vessel_prefix = cfg["prefix"]

            if cfg["v_raw_segment_input"] in h5file:
                v_raw_seg = np.asarray(h5file[cfg["v_raw_segment_input"]])
                self._pack_vessel_outputs(metrics, vessel_prefix, v_raw_seg, T)

        return ProcessResult(metrics=metrics)
