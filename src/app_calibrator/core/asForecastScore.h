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
        CRPSS, // CRPS skill score using the approximation with the rectangle method
        CRPSAR, // approximation with the rectangle method
        CRPSEP, // exact by means of primitive
        CRPSaccuracyAR, // approximation with the rectangle method
        CRPSaccuracyEP, // exact by means of primitive
        CRPSsharpnessAR, // approximation with the rectangle method
        CRPSsharpnessEP, // exact by means of primitive
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

    /** Access m_Score
     * \return The current value of m_Score
     */
    Score GetScore()
    {
        return m_Score;
    }

    /** Set m_Score
     * \param val New value to set
     */
    void SetScore(Score val)
    {
        m_Score = val;
    }

    /** Access m_Name
     * \return The current value of m_Name
     */
    wxString GetName()
    {
        return m_Name;
    }

    /** Set m_Name
     * \param val New value to set
     */
    void SetName(const wxString &val)
    {
        m_Name = val;
    }

    /** Access m_FullName
     * \return The current value of m_FullName
     */
    wxString GetFullName()
    {
        return m_FullName;
    }

    /** Set m_FullName
     * \param val New value to set
     */
    void SetFullName(const wxString &val)
    {
        m_FullName = val;
    }

    /** Access m_Order
     * \return The current value of m_Order
     */
    Order GetOrder()
    {
        return m_Order;
    }

    /** Set m_Order
     * \param val New value to set
     */
    void SetOrder(Order val)
    {
        m_Order = val;
    }

    /** Access m_ScaleBest
     * \return The current value of m_ScaleBest
     */
    float GetScaleBest()
    {
        return m_ScaleBest;
    }

    /** Set m_ScaleBest
     * \param val New value to set
     */
    void SetScaleBest(float val)
    {
        m_ScaleBest = val;
    }

    /** Access m_ScaleWorst
     * \return The current value of m_ScaleWorst
     */
    float GetScaleWorst()
    {
        return m_ScaleWorst;
    }

    /** Set m_ScaleWorst
     * \param val New value to set
     */
    void SetScaleWorst(float val)
    {
        m_ScaleWorst = val;
    }

    /** Access m_ScoreClimatology
     * \return The current value of m_ScoreClimatology
     */
    float GetScoreClimatology()
    {
        return m_ScoreClimatology;
    }

    /** Set m_ScoreClimatology
     * \param val New value to set
     */
    void SetScoreClimatology(float val)
    {
        m_ScoreClimatology = val;
    }

    /** Access m_Threshold
     * \return The current value of m_Threshold
     */
    float GetThreshold()
    {
        return m_Threshold;
    }

    /** Set m_Threshold
     * \param val New value to set
     */
    void SetThreshold(float val)
    {
        m_Threshold = val;
    }

    /** Access m_Percentile
     * \return The current value of m_Percentile
     */
    float GetPercentile()
    {
        return m_Percentile;
    }

    /** Set m_Percentile
     * \param val New value to set
     */
    void SetPercentile(float val)
    {
        m_Percentile = val;
    }

    /** Access m_Percentile
     * \return The current value of m_UsesClimatology
     */
    bool UsesClimatology()
    {
        return m_UsesClimatology;
    }



protected:
    Score m_Score; //!< Member variable "m_Score"
    wxString m_Name; //!< Member variable "m_Name"
    wxString m_FullName; //!< Member variable "m_FullName"
    Order m_Order; //!< Member variable "m_Order"
    float m_ScaleBest; //!< Member variable "m_ScaleBest"
    float m_ScaleWorst; //!< Member variable "m_ScaleWorst"
    Array1DFloat m_ArrayScoresClimatology; //!< Member variable "m_ArrayScoresClimatology"
    float m_ScoreClimatology; //!< Member variable "m_ScoreClimatology"
    bool m_UsesClimatology;
    float m_Threshold; //!< For discrete scores
    float m_Percentile; //!< For discrete scores

private:

};

#endif
