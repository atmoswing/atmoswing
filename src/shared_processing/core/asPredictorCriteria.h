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


class asDataPredictor;

class asPredictorCriteria
        : public wxObject
{
public:
    enum Criteria
    {
        Undefined,
        S1, // Teweles-Wobus
        NS1, // Normalized Teweles-Wobus
        S1grads, // Teweles-Wobus on gradients
        NS1grads, // Normalized Teweles-Wobus on gradients
        SAD, // Sum of absolute differences
        MD, // Mean difference
        NMD, // Normalized Mean difference
        MRDtoMax, // Mean Relative difference to the max value
        MRDtoMean, // Mean Relative difference to the mean value
        RMSE, // Root mean square error
        NRMSE, // Normalized Root mean square error (min-max approach)
        RMSEwithNaN, // Root mean square error with NaNs management
        RMSEonMeanWithNaN, // Root Mean Square Error on the mean value of the grid, with NaNs management
        RSE // Root square error (According to Bontron. Should not be used !)
    };

    asPredictorCriteria(int linAlgebraMethod = asLIN_ALGEBRA_NOVAR);

    static asPredictorCriteria *GetInstance(Criteria criteriaEnum, int linAlgebraMethod = asLIN_ALGEBRA_NOVAR);

    static asPredictorCriteria *GetInstance(const wxString &criteriaString, int linAlgebraMethod = asLIN_ALGEBRA_NOVAR);

    virtual ~asPredictorCriteria();

    virtual float Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb) = 0;

    bool NeedsDataRange()
    {
        return m_needsDataRange;
    }

    void SetDataRange(const asDataPredictor *data);

    void SetDataRange(float minValue, float maxValue);

    Criteria GetType()
    {
        return m_criteria;
    }

    wxString GetName()
    {
        return m_name;
    }

    wxString GetFullName()
    {
        return m_fullName;
    }

    Order GetOrder()
    {
        return m_order;
    }

    bool CanUseInline()
    {
        return m_canUseInline;
    }

protected:
    enum Criteria m_criteria;
    wxString m_name;
    wxString m_fullName;
    Order m_order;
    bool m_needsDataRange;
    float m_dataMin;
    float m_dataMax;
    float m_scaleBest;
    float m_scaleWorst;
    int m_linAlgebraMethod;
    bool m_canUseInline;

private:

};

#endif
