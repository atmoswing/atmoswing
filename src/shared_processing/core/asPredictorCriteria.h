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
 */
 
#ifndef ASPREDICTORCRITERIA_H
#define ASPREDICTORCRITERIA_H

#include <asIncludes.h>


class asPredictorCriteria: public wxObject
{
public:

    enum Criteria //!< Enumaration of managed criteria
    {
        Undefined,
        S1, // Teweles-Wobus
        S1grads, // Teweles-Wobus on gradients
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
    virtual float Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb) = 0;
    
    /** Access m_criteria
     * \return The current value of m_criteria
     */
    Criteria GetType()
    {
        return m_criteria;
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
    void SetOrder(const Order val)
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

    /** Access m_linAlgebraMethod
     * \return The current value of m_linAlgebraMethod
     */
    int GetLinAlgebraMethod()
    {
        return m_linAlgebraMethod;
    }

    /** Set m_linAlgebraMethod
     * \param val New value to set
     */
    void SetLinAlgebraMethod(int val)
    {
        m_linAlgebraMethod = val;
    }

    bool CanUseInline()
    {
        return m_canUseInline;
    }
    

protected:
    enum Criteria m_criteria; //!< Member variable "m_criteria"
    wxString m_name; //!< Member variable "m_name"
    wxString m_fullName; //!< Member variable "m_fullName"
    Order m_order; //!< Member variable "m_order"
    float m_scaleBest; //!< Member variable "m_scaleBest"
    float m_scaleWorst; //!< Member variable "m_scaleWorst"
    int m_linAlgebraMethod; //!< Member variable "m_linAlgebraMethod"
    bool m_canUseInline;

private:


};

#endif
