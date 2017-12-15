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

#ifndef ASGEOPOINT_H
#define ASGEOPOINT_H

#include <asIncludes.h>
#include <asGeo.h>

class asGeoPoint
        : public asGeo
{
public:
    asGeoPoint(const Coo &point, float level = asNONE, float height = asNONE);

    asGeoPoint(double x, double y, float level = asNONE, float height = asNONE);

    virtual ~asGeoPoint();

    Coo GetCoo()
    {
        return m_point;
    }

    void SetCoo(const Coo &val)
    {
        m_point = val;
        Init();
    }

    double GetX() const
    {
        return m_point.x;
    }

    double GetY() const
    {
        return m_point.y;
    }

    float GetLevel() const
    {
        return m_level;
    }

    void SetLevel(float val)
    {
        m_level = val;
    }

protected:

private:
    Coo m_point;
    float m_level;
    float m_height;

    void Init();

    bool DoCheckPoints();
};

#endif
