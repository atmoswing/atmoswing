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

#ifndef AS_CRITERIA_H
#define AS_CRITERIA_H

#include "asHeadersBase.h"

class asPredictor;

class asCriteria : public wxObject {
  public:
    asCriteria(const wxString& name, const wxString& fullname, Order order);

    /**
     * Factory: returns a heap-allocated criterion owned by the caller via unique_ptr.
     * On unknown criteria name, logs an error and returns nullptr.
     */
    static std::unique_ptr<asCriteria> GetInstance(const wxString& criteriaString);

    ~asCriteria() override;

    virtual float Assess(const ra2f& refData, const ra2f& evalData, int rowsNb, int colsNb) const = 0;

    void SetDataRange(const asPredictor* data);

    void SetDataRange(float minValue, float maxValue);

    void CheckNaNs(const asPredictor* ptor1, const asPredictor* ptor2);

    static a2f GetGauss2D(int nY, int nX);

    wxString GetName() const {
        return _name;
    }

    wxString GetFullName() const {
        return _fullName;
    }

    Order GetOrder() const {
        return _order;
    }

    int GetMinPointsNb() const {
        return _minPointsNb;
    }

    bool CanUseInline() const {
        return _canUseInline;
    }

    bool CheckNans() const {
        return _checkNaNs;
    }

  protected:
    wxString _name;
    wxString _fullName;
    Order _order;
    int _minPointsNb;
    float _scaleBest;
    float _scaleWorst;
    bool _canUseInline;
    bool _checkNaNs;

  private:
};

#endif
