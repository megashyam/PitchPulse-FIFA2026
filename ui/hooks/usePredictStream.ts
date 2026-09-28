/**
 * usePredictStream
 *
 * Fetches tournament prediction data and polls /predict/status while
 * a simulation is running. Auto-refreshes when a "prediction_update"
 * event is received on the existing SSE stream.
 *
 * A 202 from GET /predict/tournament (the backend started a sim because
 * none exists yet) sets isLoading and starts polling, like triggerSim().
 *
 * Usage:
 *   const { prediction, status, isLoading, error, triggerSim } = usePredictStream()
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TournamentPrediction, SimStatus } from "@/types/predict";
import { triggerHeaders } from "@/lib/api";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// How often to poll status while sim is running (ms)
const POLL_INTERVAL_MS = 2_000;

interface UsePredictStreamReturn {
    prediction: TournamentPrediction | null;
    status: SimStatus | null;
    isLoading: boolean;
    error: string | null;
    triggerSim: (nSims?: number) => Promise<void>;
    refresh: () => void;
}

export function usePredictStream(): UsePredictStreamReturn {
    const [prediction, setPrediction] = useState<TournamentPrediction | null>(null);
    const [status, setStatus] = useState<SimStatus | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const mountRef = useRef(true);
    // Refs let fetchPrediction and fetchStatus reference each other's
    // latest closures without a circular useCallback dependency chain.
    const fetchStatusRef = useRef<() => Promise<void>>(async () => {});

    const stopPolling = useCallback(() => {
        if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
        }
    }, []);

    const startPolling = useCallback(() => {
        if (pollRef.current) return;
        pollRef.current = setInterval(() => { fetchStatusRef.current(); }, POLL_INTERVAL_MS);
    }, []);

    // ------------------------------------------------------------------
    // Fetch latest prediction
    // ------------------------------------------------------------------
    const fetchPrediction = useCallback(async () => {
        try {
            const res = await fetch(`${API}/predict/tournament`);
            if (res.status === 202) {
                // Backend started a sim: show loading and start polling.
                if (mountRef.current) setIsLoading(true);
                startPolling();
                return;
            }
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            const data: TournamentPrediction = await res.json();
            if (mountRef.current) {
                setPrediction(data);
                setError(null);
            }
        } catch (err) {
            if (mountRef.current) setError(String(err));
        }
    }, [startPolling]);

    // ------------------------------------------------------------------
    // Poll status while running
    // ------------------------------------------------------------------
    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API}/predict/status`);
            if (!res.ok) return;
            const s: SimStatus = await res.json();
            if (!mountRef.current) return;
            setStatus(s);

            if (s.status === "complete") {
                stopPolling();
                setIsLoading(false);
                await fetchPrediction();
            } else if (s.status === "error") {
                stopPolling();
                setIsLoading(false);
                setError(s.error ?? "Simulation failed");
            }
        } catch {
            // ignore transient errors during polling
        }
    }, [fetchPrediction, stopPolling]);

    useEffect(() => {
        fetchStatusRef.current = fetchStatus;
    }, [fetchStatus]);

    // ------------------------------------------------------------------
    // Trigger a new simulation
    // ------------------------------------------------------------------
    const triggerSim = useCallback(async (nSims = 50_000) => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await fetch(`${API}/predict/simulate?n_sims=${nSims}`, {
                method: "POST",
                headers: triggerHeaders(),
            });
            if (res.status === 409) {
                // Already running — just start polling
                startPolling();
                return;
            }
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            startPolling();
        } catch (err) {
            setIsLoading(false);
            setError(String(err));
        }
    }, [startPolling]);

    // ------------------------------------------------------------------
    // Initial load + SSE prediction_update listener
    // ------------------------------------------------------------------
    useEffect(() => {
        mountRef.current = true;

        // Load whatever's in Redis; on a 202, fetchPrediction starts polling.
        fetchPrediction();
        fetchStatus();

        // Auto-refresh when prediction_worker (or a manual trigger) lands a new sim.
        const es = new EventSource(`${API}/predict/stream`);
        es.addEventListener("prediction_update", () => {
            if (mountRef.current) fetchPrediction();
        });
        es.onerror = () => { };

        return () => {
            mountRef.current = false;
            stopPolling();
            es.close();
        };
    }, [fetchPrediction, fetchStatus, stopPolling]);

    const refresh = useCallback(() => {
        fetchPrediction();
    }, [fetchPrediction]);

    return { prediction, status, isLoading, error, triggerSim, refresh };
}
