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

#ifndef AS_AREA_H
#define AS_AREA_H

#include "asIncludes.h"

class asArea : public wxObject {
  public:
    asArea(const Coo& cornerUL, const Coo& cornerUR, const Coo& cornerLL, const Coo& cornerLR,
           int flatAllowed = asFLAT_FORBIDDEN, bool isLatLon = true);

    asArea(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed = asFLAT_FORBIDDEN,
           bool isLatLon = true);

    asArea();

    ~asArea() override = default;

    void CheckPoint(Coo& point);

    bool IsLatLon() const {
        return m_isLatLon;
    }

    Coo GetCornerUL() const {
        return m_cornerUL;
    }

    void SetCornerUL(const Coo& val, bool noInit = false) {
        m_cornerUL = val;
        if (!noInit) Init();
    }

    Coo GetCornerUR() const {
        return m_cornerUR;
    }

    void SetCornerUR(const Coo& val, bool noInit = false) {
        m_cornerUR = val;
        if (!noInit) Init();
    }

    Coo GetCornerLL() const {
        return m_cornerLL;
    }

    void SetCornerLL(const Coo& val, bool noInit = false) {
        m_cornerLL = val;
        if (!noInit) Init();
    }

    Coo GetCornerLR() const {
        return m_cornerLR;
    }

    void SetCornerLR(const Coo& val, bool noInit = false) {
        m_cornerLR = val;
        if (!noInit) Init();
    }

    virtual double GetXmin() const;

    virtual double GetXmax() const;

    double GetXwidth() const;

    virtual double GetYmin() const;

    virtual double GetYmax() const;

    double GetYwidth() const;

    virtual bool IsRectangle() const;

    bool FlatsAllowed() const {
        return m_flatAllowed;
    }

  protected:
    Coo m_cornerUL;
    Coo m_cornerUR;
    Coo m_cornerLL;
    Coo m_cornerLR;
    int m_flatAllowed;
    bool m_isLatLon;

    virtual void Init();

  private:
    void DoCheckPoints();

    bool CheckConsistency();
};

#endif
