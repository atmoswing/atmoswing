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

#ifndef ASGEOAREA_H
#define ASGEOAREA_H

#include <asIncludes.h>
#include <asGeo.h>

class asGeoArea
        : public asGeo
{
public:
    asGeoArea(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level = asNONE,
              float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoArea(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level = asNONE, float Height = asNONE,
              int flatAllowed = asFLAT_FORBIDDEN);

    asGeoArea(float Level = asNONE, float Height = asNONE);

    virtual ~asGeoArea();

    void Generate(double Xmin, double Xwidth, double Ymin, double Ywidth, int flatAllowed = asFLAT_FORBIDDEN);

    Coo GetCornerUL()
    {
        return m_cornerUL;
    }

    void SetCornerUL(const Coo &val)
    {
        m_cornerUL = val;
        Init();
    }

    Coo GetCornerUR()
    {
        return m_cornerUR;
    }

    void SetCornerUR(const Coo &val)
    {
        m_cornerUR = val;
        Init();
    }

    Coo GetCornerLL()
    {
        return m_cornerLL;
    }

    void SetCornerLL(const Coo &val)
    {
        m_cornerLL = val;
        Init();
    }

    Coo GetCornerLR()
    {
        return m_cornerLR;
    }

    void SetCornerLR(const Coo &val)
    {
        m_cornerLR = val;
        Init();
    }

    double GetLevel()
    {
        return m_level;
    }

    void SetLevel(float val)
    {
        m_level = val;
    }

    double GetXmin();

    double GetXmax();

    double GetXwidth();

    double GetYmin();

    double GetYmax();

    double GetYwidth();

    Coo GetCenter();

    bool IsRectangle();

protected:
    Coo m_cornerUL;
    Coo m_cornerUR;
    Coo m_cornerLL;
    Coo m_cornerLR;
    float m_level;
    float m_height;
    int m_flatAllowed;

private:
    void Init();

    bool DoCheckPoints();

    bool CheckConsistency();
};

#endif
