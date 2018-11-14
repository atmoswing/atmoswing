/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ASSCORE_H
#define ASSCORE_H

#include <asIncludes.h>

class asScore
        : public wxObject
{
public:

    enum Score //!< Enumaration of scores
    {
        Undefined, CRPSS, // CRPS skill score using the approximation with the rectangle method
        CRPSAR, // approximation with the rectangle method
        CRPSEP, // exact by means of primitive
        CRPSaccuracyAR, // approximation with the rectangle method (Bontron, 2004)
        CRPSaccuracyEP, // exact by means of primitive (Bontron, 2004)
        CRPSsharpnessAR, // approximation with the rectangle method (Bontron, 2004)
        CRPSsharpnessEP, // exact by means of primitive (Bontron, 2004)
        CRPSHersbachDecomp, // Hersbach (2000) decomposition of the CRPS
        CRPSreliability, // reliability of the CRPS (Hersbach, 2000)
        CRPSpotential, // CRPS potential (Hersbach, 2000)
        DF0, // absolute difference of the frequency of null precipitations
        ContingencyTable, // Contingency table
        PC, // Proportion correct
        TS, // Threat score
        BIAS, // Bias
        FARA, // False alarm ratio
        H, // Hit rate
        F, // False alarm rate
        HSS, // Heidke skill score
        PSS, // Pierce skill score
        GSS, // Gilbert skill score
        MAE, // Mean absolute error
        RMSE, // Root mean squared error
        BS, // Brier score
        BSS, // Brier skill score
        SEEPS, // Stable equitable error in probability space
        RankHistogram, // The Verification Rank Histogram (Talagrand Diagram)
        RankHistogramReliability // Reliability of the Verification Rank Histogram (Talagrand Diagram)
    };

    asScore();

    asScore(Score score, const wxString &name, const wxString &fullname, Order order, float scaleBest = NaNf,
            float scaleWorst = NaNf, bool usesClimatology = false, bool singleValue = true);

    ~asScore() override = default;

    static asScore *GetInstance(Score scoreEnums);

    static asScore *GetInstance(const wxString &scoreString);

    virtual bool ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData) = 0;

    virtual float Assess(float observedVal, const a1f &forcastVals, int nbElements) const = 0;

    virtual a1f AssessOnArray(float observedVal, const a1f &forcastVals, int nbElements) const;

    bool CheckObservedValue(float observedVal) const;

    bool CheckVectorLength(const a1f &forcastVals, int nbElements) const;

    int CleanNans(const a1f &forcastVals, a1f &forcastValsSorted, int nbElements) const;

    wxString GetName() const
    {
        return m_name;
    }

    wxString GetFullName() const
    {
        return m_fullName;
    }

    Order GetOrder() const
    {
        return m_order;
    }

    float GetScoreClimatology() const
    {
        return m_scoreClimatology;
    }

    void SetScoreClimatology(float val)
    {
        m_scoreClimatology = val;
    }

    float GetThreshold() const
    {
        return m_threshold;
    }

    void SetThreshold(float val)
    {
        m_threshold = val;
    }

    float GetQuantile() const
    {
        return m_quantile;
    }

    void SetQuantile(float val)
    {
        m_quantile = val;
    }

    bool UsesClimatology() const
    {
        return m_usesClimatology;
    }

    bool SingleValue() const
    {
        return m_singleValue;
    }

protected:
    Score m_score;
    wxString m_name;
    wxString m_fullName;
    Order m_order;
    float m_scaleBest;
    float m_scaleWorst;
    bool m_usesClimatology;
    bool m_singleValue;
    float m_scoreClimatology;
    float m_threshold;
    float m_quantile;

private:

};

#endif
