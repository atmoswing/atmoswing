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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */
 
#ifndef ASFORECASTSCORE_H
#define ASFORECASTSCORE_H

#include <asIncludes.h>

class asForecastScore: public wxObject
{
public:

    enum Score //!< Enumaration of forecast scores
    {
        Undefined,
        CRPSS, // CRPS skill score using the approximation with the rectangle method
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
		SEEPS ,// Stable equitable error in probability space
        RankHistogram, // The Verification Rank Histogram (Talagrand Diagram)
        RankHistogramReliability // Reliability of the Verification Rank Histogram (Talagrand Diagram)
    };

    /** Default constructor
     * \param score The chosen score
     */
    asForecastScore();

    /** Default destructor */
    virtual ~asForecastScore();

    static asForecastScore* GetInstance(Score scoreEnums);

    static asForecastScore* GetInstance(const wxString& scoreString);

    /** Process the score of the climatology
     * \param score The chosen score
     * \return True on success
     */
    virtual bool ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData) = 0;

    /** Process the score
     * \param ObservedVal The observed value
     * \param ForcastVals The array of analogs values
     * \param NbElements The number of analogs to consider
     * \return The score
     */
    virtual float Assess(float ObservedVal, const Array1DFloat &ForcastVals, int NbElements) = 0;
    
    /** Process the score
     * \param ObservedVal The observed value
     * \param ForcastVals The array of analogs values
     * \param NbElements The number of analogs to consider
     * \return The score
     */
    virtual Array1DFloat AssessOnArray(float ObservedVal, const Array1DFloat &ForcastVals, int NbElements);

    /** Process some checks on the inputs
     * \param ObservedVal The observed value
     * \param ForcastVals The array of analogs values
     * \param nbElements The number of analogs to consider
     * \return True on success
     */
    bool CheckInputs(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements);

    /** Remove the NaNs from the first array and set the data in another array, which is of the desired dimension
     * \param ForcastVals The array of analogs values
     * \param ForcastValsSorted The resulting array of analogs values without NaNs and of the correct dimension
     * \param nbElements The number of analogs to keep
     * \return The resulting analogs number
     */
    int CleanNans(const Array1DFloat &ForcastVals, Array1DFloat &ForcastValsSorted, int nbElements);

    /** Access m_score
     * \return The current value of m_score
     */
    Score GetScore()
    {
        return m_score;
    }

    /** Set m_score
     * \param val New value to set
     */
    void SetScore(Score val)
    {
        m_score = val;
    }

    /** Access m_name
     * \return The current value of m_name
     */
    wxString GetName()
    {
        return m_name;
    }

    /** Set m_name
     * \param val New value to set
     */
    void SetName(const wxString &val)
    {
        m_name = val;
    }

    /** Access m_fullName
     * \return The current value of m_fullName
     */
    wxString GetFullName()
    {
        return m_fullName;
    }

    /** Set m_fullName
     * \param val New value to set
     */
    void SetFullName(const wxString &val)
    {
        m_fullName = val;
    }

    /** Access m_order
     * \return The current value of m_order
     */
    Order GetOrder()
    {
        return m_order;
    }

    /** Set m_order
     * \param val New value to set
     */
    void SetOrder(Order val)
    {
        m_order = val;
    }

    /** Access m_scaleBest
     * \return The current value of m_scaleBest
     */
    float GetScaleBest()
    {
        return m_scaleBest;
    }

    /** Set m_scaleBest
     * \param val New value to set
     */
    void SetScaleBest(float val)
    {
        m_scaleBest = val;
    }

    /** Access m_scaleWorst
     * \return The current value of m_scaleWorst
     */
    float GetScaleWorst()
    {
        return m_scaleWorst;
    }

    /** Set m_scaleWorst
     * \param val New value to set
     */
    void SetScaleWorst(float val)
    {
        m_scaleWorst = val;
    }

    /** Access m_scoreClimatology
     * \return The current value of m_scoreClimatology
     */
    float GetScoreClimatology()
    {
        return m_scoreClimatology;
    }

    /** Set m_scoreClimatology
     * \param val New value to set
     */
    void SetScoreClimatology(float val)
    {
        m_scoreClimatology = val;
    }

    /** Access m_threshold
     * \return The current value of m_threshold
     */
    float GetThreshold()
    {
        return m_threshold;
    }

    /** Set m_threshold
     * \param val New value to set
     */
    void SetThreshold(float val)
    {
        m_threshold = val;
    }

    /** Access m_quantile
     * \return The current value of m_quantile
     */
    float GetQuantile()
    {
        return m_quantile;
    }

    /** Set m_quantile
     * \param val New value to set
     */
    void SetQuantile(float val)
    {
        m_quantile = val;
    }

    /** Access m_quantile
     * \return The current value of m_usesClimatology
     */
    bool UsesClimatology()
    {
        return m_usesClimatology;
    }

    bool SingleValue()
    {
        return m_singleValue;
    }



protected:
    Score m_score; //!< Member variable "m_score"
    wxString m_name; //!< Member variable "m_name"
    wxString m_fullName; //!< Member variable "m_fullName"
    Order m_order; //!< Member variable "m_order"
    float m_scaleBest; //!< Member variable "m_scaleBest"
    float m_scaleWorst; //!< Member variable "m_scaleWorst"
    float m_scoreClimatology; //!< Member variable "m_scoreClimatology"
    bool m_usesClimatology;
    bool m_singleValue;
    float m_threshold; //!< For discrete scores
    float m_quantile; //!< For discrete scores

private:

};

#endif
