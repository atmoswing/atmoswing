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
 */
 
#ifndef ASPREDICTORCRITERIA_H
#define ASPREDICTORCRITERIA_H

#include <asIncludes.h>


class asPredictorCriteria: public wxObject
{
public:

    enum Criteria //!< Enumaration of managed criteria
    {
        S1, // Teweles-Wobus
        S1grads, // Teweles-Wobus on gradients
        S1weights, // Teweles-Wobus with axis weighting
        SAD, // Sum of absolute differences
        MD, // Mean difference
        MRDtoMax, // Mean Relative difference to the max value
        MRDtoMean, // Mean Relative difference to the mean value
        RMSE, // Root mean square error
        RMSEwithNaN, // Root mean square error with NaNs management
        RMSEonMeanWithNaN, // Root Mean Square Error on the mean value of the grid, with NaNs management
        RSE // Root square error (According to Bontron. Should not be used !)
    };

    /** Default constructor
     * \param criteria The chosen criteria
     */
    asPredictorCriteria(int linAlgebraMethod=asLIN_ALGEBRA_NOVAR);


    static asPredictorCriteria* GetInstance(Criteria criteriaEnum, int linAlgebraMethod=asLIN_ALGEBRA_NOVAR);
    static asPredictorCriteria* GetInstance(const wxString &criteriaString, int linAlgebraMethod=asLIN_ALGEBRA_NOVAR);

    /** Default destructor */
    virtual ~asPredictorCriteria();

    /** Process the Criteria
     * \param refData The target day
     * \param evalData The day to assess
     * \return The Criteria
     */
    virtual float Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb=0, int colsNb=0) = 0;

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
    void SetOrder(const Order val)
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

    /** Access m_LinAlgebraMethod
     * \return The current value of m_LinAlgebraMethod
     */
    int GetLinAlgebraMethod()
    {
        return m_LinAlgebraMethod;
    }

    /** Set m_LinAlgebraMethod
     * \param val New value to set
     */
    void SetLinAlgebraMethod(int val)
    {
        m_LinAlgebraMethod = val;
    }

    bool CanUseInline()
    {
        return m_CanUseInline;
    }
    

protected:
    enum Criteria m_Criteria; //!< Member variable "m_Criteria"
    wxString m_Name; //!< Member variable "m_Name"
    wxString m_FullName; //!< Member variable "m_FullName"
    Order m_Order; //!< Member variable "m_Order"
    float m_ScaleBest; //!< Member variable "m_ScaleBest"
    float m_ScaleWorst; //!< Member variable "m_ScaleWorst"
    int m_LinAlgebraMethod; //!< Member variable "m_LinAlgebraMethod"
    bool m_CanUseInline;

private:


};

#endif
