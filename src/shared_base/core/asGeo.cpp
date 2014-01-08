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
 
#include "asGeo.h"

asGeo::asGeo(CoordSys val)
{
    m_CoordSys = val;
    InitBounds();
}

asGeo::~asGeo()
{
    //dtor
}

void asGeo::InitBounds()
{
    switch (m_CoordSys)
        {
        case WGS84:
            m_AxisUmin = 0;
            m_AxisUmax = 360;
            m_AxisVmin = -90;
            m_AxisVmax = 90;
            break;
        default:
            m_AxisUmin = 0;
            m_AxisUmax = 0;
            m_AxisVmin = 0;
            m_AxisVmax = 0;
        }
}

bool asGeo::CheckPoint(Coo &Point, int ChangesAllowed)
{
    switch (m_CoordSys)
        {
        case WGS84:
            if(Point.v<m_AxisVmin)
            {
                if (ChangesAllowed == asEDIT_ALLOWED)
                {
                    Point.v = m_AxisVmin + (m_AxisVmin - Point.v);
                    Point.u = Point.u + 180;
                }
                return false;
            }
            if(Point.v>m_AxisVmax)
            {
                if (ChangesAllowed == asEDIT_ALLOWED)
                {
                    Point.v = m_AxisVmax + (m_AxisVmax - Point.v);
                    Point.u = Point.u + 180;
                }
                return false;
            }
            if(Point.u<m_AxisUmin)
            {
                if (ChangesAllowed == asEDIT_ALLOWED)
                {
                    Point.u += m_AxisUmax;
                }
                return false;
            }
            if(Point.u>m_AxisUmax)
            {
                if (ChangesAllowed == asEDIT_ALLOWED)
                {
                    Point.u -= m_AxisUmax;
                }
                return false;
            }
            break;
        default:
            return true;
        }
    return true;
}

wxString asGeo::GetCoordSysInfo()
{
    wxString desc = wxEmptyString;
    switch (m_CoordSys)
    {
    case WGS84:
        desc = _("WGS84: World Geodetic System 1984 (longitudes/latitudes)");
        break;
    case CH1903:
        desc = _("CH1903: The former swiss projection");
        break;
    case CH1903p:
        desc = _("CH1903+: The new swiss projection");
        break;
    default:
        asThrowException("The given coordinate system doesn't exist.");
    }

    return desc;
}

Coo asGeo::ProjTransform(CoordSys newcoordsys, Coo coo_src)
{
    Coo coo_dst;

    switch (m_CoordSys)
    {
    case WGS84:
        switch (newcoordsys)
        {
        case WGS84:
            coo_dst = coo_src;// nothing to do
            break;
        case CH1903:
            coo_dst = ProjWGS84toCH1903(coo_src);
            break;
        case CH1903p:
            coo_dst = ProjWGS84toCH1903p(coo_src);
            break;
        default:
            asThrowException(_("The destination coordinate system doesn't exist."));
        }
        break;
    case CH1903:
        switch (newcoordsys)
        {
        case WGS84:
            coo_dst = ProjCH1903toWGS84(coo_src);
            break;
        case CH1903:
            coo_dst = coo_src;// nothing to do
            break;
        case CH1903p:
            coo_dst = ProjCH1903toCH1903p(coo_src);
            break;
        default:
            asThrowException(_("The destination coordinate system doesn't exist."));
        }
        break;
    case CH1903p:
        switch (newcoordsys)
        {
        case WGS84:
            coo_dst = ProjCH1903ptoWGS84(coo_src);
            break;
        case CH1903:
            coo_dst = ProjCH1903ptoCH1903(coo_src);
            break;
        case CH1903p:
            coo_dst = coo_src;// nothing to do
            break;
        default:
            asThrowException(_("The destination coordinate system doesn't exist."));
        }
        break;
    default:
        asThrowException(_("The source coordinate system doesn't exist."));
    }

    return coo_dst;
}

Coo asGeo::ProjWGS84toCH1903(Coo coo_src)
{
    // EPSG:21781
    // http://spatialreference.org/ref/epsg/21781/
    // based on: cs2cs +proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs +to +proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs

    projPJ ref_src, ref_dst;
    projUV data;
    double height = 0;
    Coo coo_dst;

    data.u = coo_src.u;
    data.v = coo_src.v;
    data.u *= DEG_TO_RAD;
    data.v *= DEG_TO_RAD;

    if ( ! (ref_src = pj_init_plus("+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if ( ! (ref_dst = pj_init_plus("+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    // The function returns zero on success (http://linux.die.net/man/3/pj_init)
    if ( pj_transform(ref_src, ref_dst, 1, 1, &data.u, &data.v, &height) != 0) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if (data.u != HUGE_VAL)
    {
        coo_dst.u = data.u;
        coo_dst.v = data.v;
    }
    else
    {
        asThrowException(_("Projection transformation failed"));
    }

    // Memory associated with the projection is freed
    pj_free(ref_src);
    pj_free(ref_dst);

    return coo_dst;
}

Coo asGeo::ProjCH1903toWGS84(Coo coo_src)
{
    // EPSG:21781
    // http://spatialreference.org/ref/epsg/21781/
    // based on: cs2cs +proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs +to +proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs

    projPJ ref_src, ref_dst;
    projUV data;
    double height = 0;
    Coo coo_dst;

    data.u = coo_src.u;
    data.v = coo_src.v;

    if ( ! (ref_src = pj_init_plus("+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if ( ! (ref_dst = pj_init_plus("+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    // The function returns zero on success (http://linux.die.net/man/3/pj_init)
    if ( pj_transform(ref_src, ref_dst, 1, 1, &data.u, &data.v, &height) != 0) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if (data.u != HUGE_VAL)
    {
        coo_dst.u = data.u;
        coo_dst.v = data.v;
    }
    else
    {
        asThrowException(_("Projection transformation failed"));
    }

    coo_dst.u *= RAD_TO_DEG;
    coo_dst.v *= RAD_TO_DEG;

    // Memory associated with the projection is freed
    pj_free(ref_src);
    pj_free(ref_dst);

    return coo_dst;
}

Coo asGeo::ProjWGS84toCH1903p(Coo coo_src)
{
    // EPSG:2056
    // http://spatialreference.org/ref/epsg/2056/

    projPJ ref_src, ref_dst;
    projUV data;
    double height = 0;
    Coo coo_dst;

    data.u = coo_src.u;
    data.v = coo_src.v;
    data.u *= DEG_TO_RAD;
    data.v *= DEG_TO_RAD;

    if ( ! (ref_src = pj_init_plus("+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if ( ! (ref_dst = pj_init_plus("+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    // The function returns zero on success (http://linux.die.net/man/3/pj_init)
    if ( pj_transform(ref_src, ref_dst, 1, 1, &data.u, &data.v, &height) != 0) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if (data.u != HUGE_VAL)
    {
        coo_dst.u = data.u;
        coo_dst.v = data.v;
    }
    else
    {
        asThrowException(_("Projection transformation failed"));
    }

    // Memory associated with the projection is freed
    pj_free(ref_src);
    pj_free(ref_dst);

    return coo_dst;
}

Coo asGeo::ProjCH1903ptoWGS84(Coo coo_src)
{
    // EPSG:2056
    // http://spatialreference.org/ref/epsg/2056/

    projPJ ref_src, ref_dst;
    projUV data;
    double height = 0;
    Coo coo_dst;

    data.u = coo_src.u;
    data.v = coo_src.v;

    if ( ! (ref_src = pj_init_plus("+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if ( ! (ref_dst = pj_init_plus("+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    // The function returns zero on success (http://linux.die.net/man/3/pj_init)
    if ( pj_transform(ref_src, ref_dst, 1, 1, &data.u, &data.v, &height) != 0) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if (data.u != HUGE_VAL)
    {
        coo_dst.u = data.u;
        coo_dst.v = data.v;
    }
    else
    {
        asThrowException(_("Projection transformation failed"));
    }

    coo_dst.u *= RAD_TO_DEG;
    coo_dst.v *= RAD_TO_DEG;

    // Memory associated with the projection is freed
    pj_free(ref_src);
    pj_free(ref_dst);

    return coo_dst;
}

Coo asGeo::ProjCH1903ptoCH1903(Coo coo_src)
{
    // EPSG:2056 & 21781

    projPJ ref_src, ref_dst;
    projUV data;
    double height = 0;
    Coo coo_dst;

    data.u = coo_src.u;
    data.v = coo_src.v;

    if ( ! (ref_src = pj_init_plus("+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if ( ! (ref_dst = pj_init_plus("+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    // The function returns zero on success (http://linux.die.net/man/3/pj_init)
    if ( pj_transform(ref_src, ref_dst, 1, 1, &data.u, &data.v, &height) != 0) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if (data.u != HUGE_VAL)
    {
        coo_dst.u = data.u;
        coo_dst.v = data.v;
    }
    else
    {
        asThrowException(_("Projection transformation failed"));
    }

    // Memory associated with the projection is freed
    pj_free(ref_src);
    pj_free(ref_dst);

    return coo_dst;
}

Coo asGeo::ProjCH1903toCH1903p(Coo coo_src)
{
    // EPSG:2056 & 21781

    projPJ ref_src, ref_dst;
    projUV data;
    double height = 0;
    Coo coo_dst;

    data.u = coo_src.u;
    data.v = coo_src.v;

    if ( ! (ref_src = pj_init_plus("+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if ( ! (ref_dst = pj_init_plus("+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs")) ) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    // The function returns zero on success (http://linux.die.net/man/3/pj_init)
    if ( pj_transform(ref_src, ref_dst, 1, 1, &data.u, &data.v, &height) != 0) {
        wxString errnost(pj_strerrno(pj_errno), wxConvUTF8);
        asThrowException(wxString::Format(_("Projection initialization failed : %s"), errnost.c_str()));
    }

    if (data.u != HUGE_VAL)
    {
        coo_dst.u = data.u;
        coo_dst.v = data.v;
    }
    else
    {
        asThrowException(_("Projection transformation failed"));
    }

    // Memory associated with the projection is freed
    pj_free(ref_src);
    pj_free(ref_dst);

    return coo_dst;
}
