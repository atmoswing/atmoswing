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

#ifndef ASGEOAREACOMPOSITE_H
#define ASGEOAREACOMPOSITE_H

#include <asIncludes.h>
#include <asGeoArea.h>

class asGeoAreaComposite
        : public asGeo
{
public:
    asGeoAreaComposite(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                       float level = asNONE, float height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaComposite(double xMin, double xWidth, double yMin, double yWidth, float level = asNONE,
                       float height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaComposite(float level = asNONE, float height = asNONE);

    virtual ~asGeoAreaComposite();

    void Generate(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed = asFLAT_FORBIDDEN);

    Coo GetCornerUL() const
    {
        return m_cornerUL;
    }

    Coo GetCornerUR() const
    {
        return m_cornerUR;
    }

    Coo GetCornerLL() const
    {
        return m_cornerLL;
    }

    Coo GetCornerLR() const
    {
        return m_cornerLR;
    }

    double GetLevel() const
    {
        return m_level;
    }

    void SetLevel(float val)
    {
        m_level = val;
    }

    double GetAbsoluteXmin() const
    {
        return m_absoluteXmin;
    }

    double GetAbsoluteXmax() const
    {
        return m_absoluteXmax;
    }

    double GetAbsoluteYmin() const
    {
        return m_absoluteYmin;
    }

    double GetAbsoluteYmax() const
    {
        return m_absoluteYmax;
    }

    double GetAbsoluteXwidth() const
    {
        return std::abs(m_absoluteXmax - m_absoluteXmin);
    }

    double GetAbsoluteYwidth() const
    {
        return std::abs(m_absoluteYmax - m_absoluteYmin);
    }

    double GetXmin() const;

    double GetXmax() const;

    double GetYmin() const;

    double GetYmax() const;

    Coo GetCenter() const;

    int GetNbComposites() const
    {
        return (int) m_composites.size();
    }

    std::vector<asGeoArea> GetComposites() const
    {
        return m_composites;
    }

    asGeoArea GetComposite(int id) const
    {
        if (id >= m_composites.size())
            asThrowException(_("The composite area doesn't exist."));
        return m_composites[id];
    }

    bool IsRectangle() const;

protected:
    std::vector<asGeoArea> m_composites;
    float m_level;
    float m_height;

    void CreateComposites();

    bool DoCheckPoints();

    bool CheckConsistency();

private:
    Coo m_cornerUL;
    Coo m_cornerUR;
    Coo m_cornerLL;
    Coo m_cornerLR;
    int m_flatAllowed;
    double m_absoluteXmin;
    double m_absoluteXmax;
    double m_absoluteYmin;
    double m_absoluteYmax;

    void Init();
};

#endif
