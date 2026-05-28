// =============================================================================
// VFMC Risk Manager — Active/Passive & Internal/External Split Calculator
// Reads VFMC RM export .xlsx and classifies AEQ / IEQ managers.
// Requires: xlnt library (https://github.com/tfussell/xlnt)
// Compile : g++ -std=c++17 -o active_passive main.cpp -lxlnt
//           or use the accompanying CMakeLists.txt
// =============================================================================

#include <xlnt/xlnt.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ─── enums ───────────────────────────────────────────────────────────────────

enum class AssetClassTag { AEQ, IEQ, OTHER };
enum class ActivityType   { ACTIVE, PASSIVE };
enum class SourceType     { INTERNAL, EXTERNAL };

// ─── helper: to‑lower ───────────────────────────────────────────────────────

static std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

// ─── helper: case-insensitive contains ──────────────────────────────────────

static bool icontains(const std::string& haystack, const std::string& needle) {
    std::string h = to_lower(haystack);
    std::string n = to_lower(needle);
    return h.find(n) != std::string::npos;
}

// ─── helper: trim whitespace ────────────────────────────────────────────────

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ─── helper: format currency ($X,XXX.Xm) ───────────────────────────────────

static std::string fmt_money(double val_m) {
    // Format as X,XXX.X with comma separators
    bool negative = val_m < 0.0;
    double abs_val = std::abs(val_m);

    std::ostringstream raw;
    raw << std::fixed << std::setprecision(1) << abs_val;
    std::string raw_str = raw.str();

    // Split at decimal
    auto dot = raw_str.find('.');
    std::string int_part = raw_str.substr(0, dot);
    std::string dec_part = (dot != std::string::npos) ? raw_str.substr(dot) : "";

    // Insert commas
    std::string with_commas;
    int count = 0;
    for (int i = static_cast<int>(int_part.size()) - 1; i >= 0; --i) {
        if (count > 0 && count % 3 == 0) with_commas.insert(with_commas.begin(), ',');
        with_commas.insert(with_commas.begin(), int_part[i]);
        ++count;
    }

    std::string result = "$" + (negative ? "-" : "") + with_commas + dec_part + "m";
    return result;
}

// ─── helper: format percentage ──────────────────────────────────────────────

static std::string fmt_pct(double pct) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << pct << "%";
    return oss.str();
}

// ─── parsed row ─────────────────────────────────────────────────────────────

struct ParsedRow {
    std::string entity_name;
    std::string asset_class_raw;
    double      exposure = 0.0;      // raw AUD value
    std::string mandate_code;
    std::string portfolio;
    std::string security_type;
};

// ─── classified manager ─────────────────────────────────────────────────────

struct ManagerEntry {
    std::string   name;
    std::string   mandate_code;
    AssetClassTag ac   = AssetClassTag::OTHER;
    ActivityType  act  = ActivityType::ACTIVE;
    SourceType    src  = SourceType::EXTERNAL;
    double        exposure_aud = 0.0;   // raw AUD
    bool          classified   = true;
};

// ─── flexible column detection ──────────────────────────────────────────────

struct ColumnMap {
    int entity      = -1;
    int asset_class  = -1;
    int exposure     = -1;
    int mandate      = -1;
    int portfolio    = -1;
    int security_type = -1;
};

static bool col_match(const std::string& header, const std::vector<std::string>& patterns) {
    std::string h = to_lower(trim(header));
    for (auto& p : patterns) {
        if (h.find(to_lower(p)) != std::string::npos) return true;
    }
    return false;
}

static ColumnMap detect_columns(const std::vector<std::string>& headers) {
    ColumnMap cm;
    for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
        if (cm.entity == -1 &&
            col_match(headers[i], {"entity_long_name", "entitylongname", "entity long name",
                                    "manager_name", "managername", "manager name"}))
            cm.entity = i;

        if (cm.asset_class == -1 &&
            col_match(headers[i], {"asset_class", "assetclass", "asset class",
                                    "business_class_level_4", "businessclasslevel4"}))
            cm.asset_class = i;

        if (cm.exposure == -1 &&
            col_match(headers[i], {"exposure_pc", "eff_exp_value_base",
                                    "effectivedateexposure", "effective_date_exposure",
                                    "exposure", "eff_exp", "mkt_val", "market_value",
                                    "marketvalue"}))
            cm.exposure = i;

        if (cm.mandate == -1 &&
            col_match(headers[i], {"scd_mandate", "mandate_code", "mandatecode",
                                    "mandate code", "mandate"}))
            cm.mandate = i;

        if (cm.portfolio == -1 &&
            col_match(headers[i], {"portfolio", "trustname", "trust_name", "trust name",
                                    "fund_name", "fundname"}))
            cm.portfolio = i;

        if (cm.security_type == -1 &&
            col_match(headers[i], {"security_type", "securitytype", "security type",
                                    "sec_type"}))
            cm.security_type = i;
    }
    return cm;
}

// ─── asset-class tagging ────────────────────────────────────────────────────

static AssetClassTag tag_asset_class(const std::string& raw) {
    if (icontains(raw, "Australian Equities") || icontains(raw, "Aust Equities") ||
        icontains(raw, "Domestic Equities") || icontains(raw, "AEQ"))
        return AssetClassTag::AEQ;

    if (icontains(raw, "International Equities") || icontains(raw, "Intl Equities") ||
        icontains(raw, "Emerging Market") || icontains(raw, "IEQ") ||
        icontains(raw, "EMT") || icontains(raw, "Low Volatility Equit"))
        return AssetClassTag::IEQ;

    return AssetClassTag::OTHER;
}

// ─── cash / residual exclusion ──────────────────────────────────────────────

static bool is_cash_residual(const ParsedRow& row) {
    if (icontains(row.security_type, "CASH BUCKET")) return true;
    if (icontains(row.security_type, "CB TRADING"))  return true;
    if (icontains(row.entity_name, "Cash Bucket"))   return true;
    if (icontains(row.entity_name, "CB Trading"))    return true;
    if (icontains(row.entity_name, "FX Residual"))   return true;
    if (icontains(row.entity_name, "Cash Residual")) return true;
    return false;
}

// ─── AEQ classification ────────────────────────────────────────────────────

static bool classify_aeq(const std::string& entity, const std::string& mandate,
                          ActivityType& act, SourceType& src) {
    // ACTIVE + EXTERNAL
    if (icontains(entity, "Cooper Investors") || icontains(mandate, "Cooper")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Greencape") || icontains(mandate, "Greencape")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Vinva") || icontains(mandate, "Vinva")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Alphinity") || icontains(mandate, "Alphinity")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Platypus") || icontains(mandate, "Platypus")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Yarra") || icontains(mandate, "Yarra")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "IFM") || icontains(mandate, "IFM")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Paradice") || icontains(mandate, "Paradice")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }

    // ACTIVE + INTERNAL
    if (icontains(entity, "OBI") || icontains(entity, "Opportunistic") ||
        icontains(mandate, "OBI") || icontains(mandate, "Opportunistic")) {
        act = ActivityType::ACTIVE; src = SourceType::INTERNAL; return true;
    }
    if (icontains(entity, "VADER") || icontains(mandate, "VADER")) {
        act = ActivityType::ACTIVE; src = SourceType::INTERNAL; return true;
    }
    if (icontains(mandate, "IMP") || icontains(mandate, "Imp")) {
        // Implementation account — internal active
        act = ActivityType::ACTIVE; src = SourceType::INTERNAL; return true;
    }

    // PASSIVE + EXTERNAL
    if (icontains(entity, "State Street") || icontains(entity, "SSgA") ||
        icontains(entity, "SSGA") || icontains(mandate, "SSgA") ||
        icontains(mandate, "SSGA") || icontains(mandate, "State Street")) {
        act = ActivityType::PASSIVE; src = SourceType::EXTERNAL; return true;
    }

    // PASSIVE + INTERNAL
    if (icontains(entity, "ASX20") || icontains(entity, "ASX 20") ||
        icontains(entity, "Plug")  || icontains(mandate, "ASX20") ||
        icontains(mandate, "ASX 20") || icontains(mandate, "Plug")) {
        act = ActivityType::PASSIVE; src = SourceType::INTERNAL; return true;
    }

    return false; // unclassified
}

// ─── IEQ classification ────────────────────────────────────────────────────

static bool classify_ieq(const std::string& entity, const std::string& mandate,
                          ActivityType& act, SourceType& src) {
    // ACTIVE + EXTERNAL
    if (icontains(entity, "Arrowstreet") || icontains(mandate, "Arrowstreet")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Wellington") || icontains(mandate, "Wellington")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Sanders") || icontains(mandate, "Sanders")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "C Worldwide") || icontains(entity, "C WorldWide") ||
        icontains(mandate, "C Worldwide") || icontains(mandate, "CWW")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Orbis") || icontains(mandate, "Orbis")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Artisan") || icontains(mandate, "Artisan")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Wasatch") || icontains(mandate, "Wasatch")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "RWC") || icontains(mandate, "RWC")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "Jennison") || icontains(mandate, "Jennison")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }
    if (icontains(entity, "GSAM") || icontains(entity, "Goldman Sachs") ||
        icontains(mandate, "GSAM") || icontains(mandate, "Goldman")) {
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }

    // ACTIVE + INTERNAL
    if (icontains(entity, "ELVIS") || icontains(entity, "ElVIS") ||
        icontains(mandate, "ELVIS") || icontains(mandate, "ElVIS")) {
        act = ActivityType::ACTIVE; src = SourceType::INTERNAL; return true;
    }
    if (icontains(entity, "Quality Plus") || icontains(entity, "QualityPlus") ||
        icontains(mandate, "QLPL") || icontains(mandate, "Quality Plus") ||
        icontains(mandate, "QualityPlus")) {
        act = ActivityType::ACTIVE; src = SourceType::INTERNAL; return true;
    }
    if (icontains(entity, "Nvidia") || icontains(entity, "NVDA") ||
        icontains(mandate, "Nvidia") || icontains(mandate, "NVDA")) {
        act = ActivityType::ACTIVE; src = SourceType::INTERNAL; return true;
    }
    // "Plug" in IEQ context → internal active (Nvidia Plug)
    if (icontains(entity, "Plug") || icontains(mandate, "Plug")) {
        act = ActivityType::ACTIVE; src = SourceType::INTERNAL; return true;
    }

    // PASSIVE + EXTERNAL — SSgA Low Carbon mandates
    if (icontains(entity, "State Street") || icontains(entity, "SSgA") ||
        icontains(entity, "SSGA") || icontains(mandate, "SSgA") ||
        icontains(mandate, "SSGA") || icontains(mandate, "State Street") ||
        icontains(entity, "Low Carbon") || icontains(mandate, "Low Carbon") ||
        icontains(mandate, "LCHOSG") || icontains(mandate, "LCWASG") ||
        icontains(entity, "Optimised") || icontains(mandate, "Optimised")) {
        act = ActivityType::PASSIVE; src = SourceType::EXTERNAL; return true;
    }

    // PASSIVE + INTERNAL — TRS / Swaps
    if (icontains(entity, "Total Return Swap") || icontains(entity, "TRS") ||
        icontains(entity, "Swap") || icontains(mandate, "SWAP") ||
        icontains(mandate, "TRS") || icontains(mandate, "MinVol") ||
        icontains(mandate, "MSVL") || icontains(entity, "Low Vol") ||
        icontains(entity, "LowVol") || icontains(entity, "Minimum Volatility") ||
        icontains(mandate, "Low Vol") || icontains(mandate, "LowVol")) {
        act = ActivityType::PASSIVE; src = SourceType::INTERNAL; return true;
    }

    // IFM anomaly in IEQ
    if (icontains(entity, "IFM") || icontains(mandate, "IFM")) {
        std::cerr << "  [WARNING] IFM Investors found in IEQ context — possible data anomaly.\n";
        act = ActivityType::ACTIVE; src = SourceType::EXTERNAL; return true;
    }

    return false; // unclassified
}

// ─── print helpers ──────────────────────────────────────────────────────────

static void print_separator(int width = 85) {
    std::cout << std::string(width, '-') << "\n";
}

static void print_asset_class_table(const std::string& label, const std::string& date_str,
                                     std::vector<ManagerEntry>& entries) {
    // Sort by exposure descending
    std::sort(entries.begin(), entries.end(),
              [](const ManagerEntry& a, const ManagerEntry& b) {
                  return a.exposure_aud > b.exposure_aud;
              });

    double total = 0.0;
    for (auto& e : entries) total += e.exposure_aud;
    double total_m = total / 1e6;

    std::cout << "\n";
    print_separator(90);
    std::cout << "  " << label << " as at " << date_str << "\n";
    print_separator(90);

    // Header
    std::cout << std::left << std::setw(32) << "Manager Name"
              << " | " << std::right << std::setw(14) << "Exposure ($m)"
              << " | " << std::left  << std::setw(14) << "Active/Passive"
              << " | " << std::left  << std::setw(14) << "Int/Ext"
              << "\n";
    print_separator(90);

    std::vector<ManagerEntry> unclassified;

    for (auto& e : entries) {
        double exp_m = e.exposure_aud / 1e6;
        std::string act_str = (e.act == ActivityType::ACTIVE) ? "Active" : "Passive";
        std::string src_str = (e.src == SourceType::INTERNAL) ? "Internal" : "External";
        std::string tag = e.classified ? "" : " [?]";

        std::cout << std::left  << std::setw(32) << (e.name + tag)
                  << " | " << std::right << std::setw(14) << fmt_money(exp_m)
                  << " | " << std::left  << std::setw(14) << act_str
                  << " | " << std::left  << std::setw(14) << src_str
                  << "\n";

        if (!e.classified) unclassified.push_back(e);
    }

    print_separator(90);
    std::cout << std::left  << std::setw(32) << ("TOTAL " + label)
              << " | " << std::right << std::setw(14) << fmt_money(total_m)
              << " |" << std::string(32, ' ') << "\n";
    print_separator(90);

    // Summaries
    double active_sum = 0, passive_sum = 0, internal_sum = 0, external_sum = 0;
    for (auto& e : entries) {
        if (e.act == ActivityType::ACTIVE)   active_sum   += e.exposure_aud;
        else                                 passive_sum  += e.exposure_aud;
        if (e.src == SourceType::INTERNAL)   internal_sum += e.exposure_aud;
        else                                 external_sum += e.exposure_aud;
    }

    auto pct = [&](double v) -> double {
        return (total > 0) ? (v / total * 100.0) : 0.0;
    };

    std::cout << "\n  SUMMARY — " << label << ":\n";
    std::cout << "    Active:    " << fmt_money(active_sum / 1e6)
              << "  (" << fmt_pct(pct(active_sum)) << ")\n";
    std::cout << "    Passive:   " << fmt_money(passive_sum / 1e6)
              << "  (" << fmt_pct(pct(passive_sum)) << ")\n";
    std::cout << "    Internal:  " << fmt_money(internal_sum / 1e6)
              << "  (" << fmt_pct(pct(internal_sum)) << ")\n";
    std::cout << "    External:  " << fmt_money(external_sum / 1e6)
              << "  (" << fmt_pct(pct(external_sum)) << ")\n";

    // 2x2 matrix
    double ai = 0, ae = 0, pi = 0, pe = 0;
    for (auto& e : entries) {
        if (e.act == ActivityType::ACTIVE && e.src == SourceType::INTERNAL)  ai += e.exposure_aud;
        if (e.act == ActivityType::ACTIVE && e.src == SourceType::EXTERNAL)  ae += e.exposure_aud;
        if (e.act == ActivityType::PASSIVE && e.src == SourceType::INTERNAL) pi += e.exposure_aud;
        if (e.act == ActivityType::PASSIVE && e.src == SourceType::EXTERNAL) pe += e.exposure_aud;
    }

    std::cout << "\n  2x2 Matrix — " << label << ":\n";
    std::cout << "                    | " << std::setw(16) << "Internal"
              << " | " << std::setw(16) << "External"
              << " | " << std::setw(16) << "Total" << "\n";
    std::cout << "  " << std::string(72, '-') << "\n";

    auto cell = [&](double v) -> std::string {
        return fmt_money(v / 1e6) + " (" + fmt_pct(pct(v)) + ")";
    };

    std::cout << "  " << std::left << std::setw(18) << "Active"
              << "  | " << std::setw(16) << cell(ai)
              << " | " << std::setw(16) << cell(ae)
              << " | " << std::setw(16) << cell(ai + ae) << "\n";
    std::cout << "  " << std::left << std::setw(18) << "Passive"
              << "  | " << std::setw(16) << cell(pi)
              << " | " << std::setw(16) << cell(pe)
              << " | " << std::setw(16) << cell(pi + pe) << "\n";
    std::cout << "  " << std::left << std::setw(18) << "Total"
              << "  | " << std::setw(16) << cell(ai + pi)
              << " | " << std::setw(16) << cell(ae + pe)
              << " | " << std::setw(16)
              << (fmt_money(total_m) + " (100.0%)") << "\n";

    // Unclassified warnings
    if (!unclassified.empty()) {
        std::cout << "\n  [!] UNCLASSIFIED MANAGERS (" << label << "):\n";
        for (auto& u : unclassified) {
            std::cout << "      • " << u.name
                      << "  (mandate: " << u.mandate_code
                      << ", exposure: " << fmt_money(u.exposure_aud / 1e6) << ")\n";
        }
    }
}

// ─── validation ─────────────────────────────────────────────────────────────

static void run_validation(const std::string& label,
                            const std::vector<ManagerEntry>& entries,
                            double expected_total_low_bn, double expected_total_high_bn,
                            double expected_passive_low_pct, double expected_passive_high_pct) {
    double total = 0, active_sum = 0, passive_sum = 0, internal_sum = 0, external_sum = 0;
    bool has_unclassified = false;
    std::map<std::string, double> mgr_exposure;

    for (auto& e : entries) {
        total += e.exposure_aud;
        if (e.act == ActivityType::ACTIVE)   active_sum   += e.exposure_aud;
        else                                 passive_sum  += e.exposure_aud;
        if (e.src == SourceType::INTERNAL)   internal_sum += e.exposure_aud;
        else                                 external_sum += e.exposure_aud;
        if (!e.classified) has_unclassified = true;
        mgr_exposure[e.name] += e.exposure_aud;
    }

    double total_bn = total / 1e9;
    double passive_pct = (total > 0) ? (passive_sum / total * 100.0) : 0.0;
    double ie_diff = std::abs((internal_sum + external_sum) - total);
    double ap_diff = std::abs((active_sum + passive_sum) - total);

    std::cout << "\n  VALIDATION — " << label << ":\n";

    // Passive %
    auto check = [](bool ok, const std::string& msg) {
        std::cout << "    " << (ok ? "[PASS]" : "[WARN]") << " " << msg << "\n";
    };

    check(passive_pct >= expected_passive_low_pct && passive_pct <= expected_passive_high_pct,
          label + " passive = " + fmt_pct(passive_pct) +
          " (expected " + fmt_pct(expected_passive_low_pct) + "-" +
          fmt_pct(expected_passive_high_pct) + ")");

    check(total_bn >= expected_total_low_bn && total_bn <= expected_total_high_bn,
          "Total " + label + " = $" +
          std::to_string(total_bn).substr(0, std::to_string(total_bn).find('.') + 2) +
          "bn (expected $" + std::to_string(expected_total_low_bn).substr(0, 2) + "-" +
          std::to_string(expected_total_high_bn).substr(0, 2) + "bn)");

    // Single manager concentration
    bool conc_ok = true;
    for (auto& [name, exp] : mgr_exposure) {
        double pct = (total > 0) ? (exp / total * 100.0) : 0.0;
        if (pct > 25.0) {
            check(false, name + " = " + fmt_pct(pct) + " (exceeds 25% threshold)");
            conc_ok = false;
        }
    }
    if (conc_ok) check(true, "No single manager exceeds 25% of " + label);

    check(!has_unclassified, "All rows classified (no unclassified managers)");
    check(ie_diff < 1.0, "Sum of Internal + External = Total (diff: $" +
          std::to_string(ie_diff / 1e6).substr(0, 6) + "m)");
    check(ap_diff < 1.0, "Sum of Active + Passive = Total (diff: $" +
          std::to_string(ap_diff / 1e6).substr(0, 6) + "m)");
}

// ─── CSV export ─────────────────────────────────────────────────────────────

static void export_csv(const std::string& filepath,
                        const std::vector<ManagerEntry>& aeq,
                        const std::vector<ManagerEntry>& ieq,
                        const std::string& date_str) {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Cannot open CSV file for writing: " << filepath << "\n";
        return;
    }

    ofs << "AssetClass,ManagerName,MandateCode,ExposureAUD,ExposureMillions,"
        << "ActivePassive,InternalExternal,Classified,Date\n";

    auto write_entries = [&](const std::vector<ManagerEntry>& entries, const std::string& ac) {
        for (auto& e : entries) {
            std::string act_str = (e.act == ActivityType::ACTIVE) ? "Active" : "Passive";
            std::string src_str = (e.src == SourceType::INTERNAL) ? "Internal" : "External";

            // Escape name for CSV
            std::string safe_name = e.name;
            if (safe_name.find(',') != std::string::npos ||
                safe_name.find('"') != std::string::npos) {
                // Double any quotes, then wrap in quotes
                std::string escaped;
                for (char c : safe_name) {
                    if (c == '"') escaped += '"';
                    escaped += c;
                }
                safe_name = "\"" + escaped + "\"";
            }

            ofs << ac << ","
                << safe_name << ","
                << e.mandate_code << ","
                << std::fixed << std::setprecision(2) << e.exposure_aud << ","
                << std::fixed << std::setprecision(1) << (e.exposure_aud / 1e6) << ","
                << act_str << ","
                << src_str << ","
                << (e.classified ? "Y" : "N") << ","
                << date_str << "\n";
        }
    };

    write_entries(aeq, "AEQ");
    write_entries(ieq, "IEQ");

    ofs.close();
    std::cout << "\n[CSV] Results exported to: " << filepath << "\n";
}

// ─── help text ──────────────────────────────────────────────────────────────

static void print_help(const char* prog) {
    std::cout << "VFMC Risk Manager — Active/Passive & Internal/External Split Calculator\n\n"
              << "Usage:\n"
              << "  " << prog << " <path_to_xlsx> [YYYYMMDD] [options]\n\n"
              << "Arguments:\n"
              << "  <path_to_xlsx>   Path to the VFMC RM export .xlsx file\n"
              << "  [YYYYMMDD]       Optional date label for the report\n\n"
              << "Options:\n"
              << "  --help           Show this help message\n"
              << "  --csv <file>     Export results to CSV file\n\n"
              << "Example:\n"
              << "  " << prog << " VFMC.Monthly.20260430.CLIENT_DTF-Act3.AssetClassExPos.xlsx 20260430\n"
              << "  " << prog << " data.xlsx 20260430 --csv output.csv\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    // ── argument parsing ────────────────────────────────────────────────
    std::string xlsx_path;
    std::string date_str = "UNKNOWN";
    std::string csv_path;
    bool do_csv = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        }
        if (arg == "--csv") {
            do_csv = true;
            if (i + 1 < argc) {
                csv_path = argv[++i];
            } else {
                std::cerr << "[ERROR] --csv requires a filename argument.\n";
                return 1;
            }
            continue;
        }
        if (xlsx_path.empty()) {
            xlsx_path = arg;
        } else if (date_str == "UNKNOWN") {
            date_str = arg;
        }
    }

    if (xlsx_path.empty()) {
        std::cerr << "[ERROR] No input file specified. Use --help for usage.\n";
        return 1;
    }

    if (!std::filesystem::exists(xlsx_path)) {
        std::cerr << "[ERROR] File not found: " << xlsx_path << "\n";
        return 1;
    }

    // Try to extract date from filename if not provided
    if (date_str == "UNKNOWN") {
        // Look for YYYYMMDD pattern in filename
        std::string fname = std::filesystem::path(xlsx_path).filename().string();
        for (size_t i = 0; i + 8 <= fname.size(); ++i) {
            bool all_digit = true;
            for (int j = 0; j < 8; ++j) {
                if (!std::isdigit(static_cast<unsigned char>(fname[i + j]))) {
                    all_digit = false;
                    break;
                }
            }
            if (all_digit) {
                date_str = fname.substr(i, 8);
                break;
            }
        }
    }

    // ── read Excel file ─────────────────────────────────────────────────
    std::cout << "[INFO] Reading: " << xlsx_path << "\n";
    std::cout << "[INFO] Date:    " << date_str << "\n\n";

    xlnt::workbook wb;
    try {
        wb.load(xlsx_path);
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Failed to open xlsx file: " << ex.what() << "\n";
        return 1;
    }

    xlnt::worksheet ws = wb.active_sheet();

    // ── detect columns from header row ──────────────────────────────────
    auto rows = ws.rows(false);
    if (ws.highest_row() < 2) {
        std::cerr << "[ERROR] File appears to have fewer than 2 rows.\n";
        return 1;
    }

    // Read header row
    std::vector<std::string> headers;
    auto header_row = ws.rows(false)[0]; // first row
    for (auto& cell : header_row) {
        headers.push_back(trim(cell.to_string()));
    }

    ColumnMap cm = detect_columns(headers);

    // Validate minimum required columns
    if (cm.entity == -1 && cm.mandate == -1) {
        std::cerr << "[ERROR] Cannot find entity/manager name column.\n";
        std::cerr << "        Headers found: ";
        for (auto& h : headers) std::cerr << "\"" << h << "\" ";
        std::cerr << "\n";
        return 1;
    }
    if (cm.asset_class == -1) {
        std::cerr << "[ERROR] Cannot find asset class column.\n";
        std::cerr << "        Headers found: ";
        for (auto& h : headers) std::cerr << "\"" << h << "\" ";
        std::cerr << "\n";
        return 1;
    }
    if (cm.exposure == -1) {
        std::cerr << "[ERROR] Cannot find exposure/market value column.\n";
        std::cerr << "        Headers found: ";
        for (auto& h : headers) std::cerr << "\"" << h << "\" ";
        std::cerr << "\n";
        return 1;
    }

    std::cout << "[INFO] Column mapping:\n";
    if (cm.entity >= 0)        std::cout << "  Entity:        col " << cm.entity << " (" << headers[cm.entity] << ")\n";
    if (cm.asset_class >= 0)   std::cout << "  Asset Class:   col " << cm.asset_class << " (" << headers[cm.asset_class] << ")\n";
    if (cm.exposure >= 0)      std::cout << "  Exposure:      col " << cm.exposure << " (" << headers[cm.exposure] << ")\n";
    if (cm.mandate >= 0)       std::cout << "  Mandate:       col " << cm.mandate << " (" << headers[cm.mandate] << ")\n";
    if (cm.portfolio >= 0)     std::cout << "  Portfolio:      col " << cm.portfolio << " (" << headers[cm.portfolio] << ")\n";
    if (cm.security_type >= 0) std::cout << "  Security Type: col " << cm.security_type << " (" << headers[cm.security_type] << ")\n";
    std::cout << "\n";

    // ── parse data rows ─────────────────────────────────────────────────
    std::vector<ParsedRow> parsed;
    int skipped_cash = 0, skipped_zero = 0, skipped_other = 0;
    int total_rows = 0;

    auto all_rows = ws.rows(false);
    for (int r = 1; r < static_cast<int>(ws.highest_row()); ++r) {
        ++total_rows;
        auto row = all_rows[r];

        auto safe_cell = [&](int col) -> std::string {
            if (col < 0) return "";
            try {
                return trim(row[col].to_string());
            } catch (...) {
                return "";
            }
        };

        auto safe_number = [&](int col) -> double {
            if (col < 0) return 0.0;
            try {
                std::string val = trim(row[col].to_string());
                if (val.empty()) return 0.0;
                // Remove commas if present
                val.erase(std::remove(val.begin(), val.end(), ','), val.end());
                return std::stod(val);
            } catch (...) {
                return 0.0;
            }
        };

        ParsedRow pr;
        pr.entity_name   = safe_cell(cm.entity);
        pr.asset_class_raw = safe_cell(cm.asset_class);
        pr.exposure      = safe_number(cm.exposure);
        pr.mandate_code  = safe_cell(cm.mandate);
        pr.portfolio     = safe_cell(cm.portfolio);
        pr.security_type = safe_cell(cm.security_type);

        // Tag asset class
        AssetClassTag tag = tag_asset_class(pr.asset_class_raw);
        if (tag == AssetClassTag::OTHER) { ++skipped_other; continue; }

        // Skip zero exposure
        if (std::abs(pr.exposure) < 0.01) { ++skipped_zero; continue; }

        // Skip cash/FX residuals
        if (is_cash_residual(pr)) { ++skipped_cash; continue; }

        // Use mandate code as fallback name
        std::string name = pr.entity_name.empty() ? pr.mandate_code : pr.entity_name;
        if (name.empty()) name = "UNKNOWN";

        // Store with tag
        ParsedRow tagged = pr;
        // We'll store tag alongside — use a small struct wrapper below
        // For now, add to a tagged list
        parsed.push_back(pr);
    }

    std::cout << "[INFO] Rows read:    " << total_rows << "\n";
    std::cout << "[INFO] Skipped (other AC): " << skipped_other << "\n";
    std::cout << "[INFO] Skipped (zero exp): " << skipped_zero << "\n";
    std::cout << "[INFO] Skipped (cash/FX):  " << skipped_cash << "\n";
    std::cout << "[INFO] Relevant rows:      " << parsed.size() << "\n";

    // ── aggregate by manager ────────────────────────────────────────────
    // Key: (AssetClassTag, manager_name) → aggregated ManagerEntry
    struct AggKey {
        AssetClassTag ac;
        std::string   name;
        std::string   mandate;
        bool operator<(const AggKey& o) const {
            if (ac != o.ac) return ac < o.ac;
            return name < o.name;
        }
    };

    std::map<AggKey, ManagerEntry> agg;

    for (auto& pr : parsed) {
        AssetClassTag ac = tag_asset_class(pr.asset_class_raw);
        std::string name = pr.entity_name.empty() ? pr.mandate_code : pr.entity_name;
        if (name.empty()) name = "UNKNOWN";

        AggKey key{ac, name, pr.mandate_code};

        auto it = agg.find(key);
        if (it == agg.end()) {
            ManagerEntry me;
            me.name = name;
            me.mandate_code = pr.mandate_code;
            me.ac = ac;
            me.exposure_aud = pr.exposure;
            agg[key] = me;
        } else {
            it->second.exposure_aud += pr.exposure;
        }
    }

    // ── classify each aggregated manager ────────────────────────────────
    std::vector<ManagerEntry> aeq_entries, ieq_entries;
    int unclassified_count = 0;

    for (auto& [key, me] : agg) {
        ActivityType act = ActivityType::ACTIVE;
        SourceType   src = SourceType::EXTERNAL;
        bool classified = false;

        if (key.ac == AssetClassTag::AEQ) {
            classified = classify_aeq(me.name, me.mandate_code, act, src);
        } else if (key.ac == AssetClassTag::IEQ) {
            classified = classify_ieq(me.name, me.mandate_code, act, src);
        }

        me.act = act;
        me.src = src;
        me.classified = classified;

        if (!classified) {
            ++unclassified_count;
            std::cerr << "  [WARNING] Unclassified: \"" << me.name
                      << "\" (mandate: " << me.mandate_code
                      << ", exposure: " << fmt_money(me.exposure_aud / 1e6)
                      << ", AC: " << (key.ac == AssetClassTag::AEQ ? "AEQ" : "IEQ")
                      << ") → defaulting to Active+External\n";
        }

        if (key.ac == AssetClassTag::AEQ) aeq_entries.push_back(me);
        else                               ieq_entries.push_back(me);
    }

    // ── output ──────────────────────────────────────────────────────────
    if (!aeq_entries.empty()) {
        print_asset_class_table("AEQ (Australian Equities)", date_str, aeq_entries);
        run_validation("AEQ", aeq_entries, 12.0, 14.0, 20.0, 35.0);
    } else {
        std::cout << "\n[INFO] No AEQ data found in file.\n";
    }

    if (!ieq_entries.empty()) {
        print_asset_class_table("IEQ (International Equities)", date_str, ieq_entries);
        run_validation("IEQ", ieq_entries, 24.0, 28.0, 22.0, 38.0);
    } else {
        std::cout << "\n[INFO] No IEQ data found in file.\n";
    }

    // ── CSV export ──────────────────────────────────────────────────────
    if (do_csv) {
        export_csv(csv_path, aeq_entries, ieq_entries, date_str);
    }

    // ── final summary ───────────────────────────────────────────────────
    std::cout << "\n";
    print_separator(90);
    std::cout << "  DONE. Managers classified: "
              << (aeq_entries.size() + ieq_entries.size() - unclassified_count)
              << " / " << (aeq_entries.size() + ieq_entries.size())
              << " | Unclassified: " << unclassified_count << "\n";
    print_separator(90);

    return 0;
}
