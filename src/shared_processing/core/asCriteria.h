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


class asPredictor;

class asCriteria
        : public wxObject
{
public:

    asCriteria(const wxString &name, const wxString &fullname, Order order);

    static asCriteria *GetInstance(const wxString &criteriaString);

    ~asCriteria() override;

    virtual float Assess(const a2f &refData, const a2f &evalData, int rowsNb, int colsNb) const = 0;

    bool NeedsDataRange() const
    {
        return m_needsDataRange;
    }

    void SetDataRange(const asPredictor *data);

    void SetDataRange(float minValue, float maxValue);

    void CheckNaNs(const asPredictor *ptor1, const asPredictor *ptor2);

    static a2f GetGauss2D(int nY, int nX);

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

    int GetMinPointsNb() const
    {
        return m_minPointsNb;
    }

    bool CanUseInline() const
    {
        return m_canUseInline;
    }

    bool CheckNans() const
    {
        return m_checkNaNs;
    }

protected:
    wxString m_name;
    wxString m_fullName;
    Order m_order;
    int m_minPointsNb;
    bool m_needsDataRange;
    float m_dataMin;
    float m_dataMax;
    float m_scaleBest;
    float m_scaleWorst;
    bool m_canUseInline;
    bool m_checkNaNs;

private:

};

#endif
