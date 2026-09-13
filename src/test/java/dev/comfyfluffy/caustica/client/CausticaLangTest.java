package dev.comfyfluffy.caustica.client;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Language file invariants for the RT options. A missing key does not crash anything — Minecraft falls back to
 * English — so nothing else would notice a locale that quietly lost an option, and a typo in a key name is
 * invisible until someone plays in that language. The options are all translated rather than left to the
 * English fallback, so a new option is not done until every locale carries it.
 */
final class CausticaLangTest {
    private static final String EN_US = "en_us";
    private static final List<String> LOCALES = List.of(
            "de_de", "en_us", "es_es", "fr_fr", "it_it", "ja_jp", "ko_kr", "pt_br", "ru_ru", "zh_cn", "zh_tw");

    /** Every locale carries a non-blank translation of every option string English defines. */
    @Test
    void everyLocaleTranslatesEveryOption() {
        Map<String, String> english = read(EN_US);
        assertFalse(english.isEmpty(), "the reference locale defines the option strings");

        for (String locale : LOCALES) {
            Map<String, String> strings = read(locale);
            List<String> missing = new ArrayList<>();
            for (String key : english.keySet()) {
                String value = strings.get(key);
                if (value == null || value.isBlank()) {
                    missing.add(key);
                }
            }
            assertTrue(missing.isEmpty(), locale + " does not translate: " + missing);
        }
    }

    /**
     * No locale may carry a key English does not have: that is either a typo (the string would never be shown)
     * or a leftover from a removed option.
     */
    @Test
    void noLocaleInventsKeysEnglishDoesNotHave() {
        Map<String, String> english = read(EN_US);
        for (String locale : LOCALES) {
            List<String> extra = new ArrayList<>();
            for (String key : read(locale).keySet()) {
                if (!english.containsKey(key)) {
                    extra.add(key);
                }
            }
            assertTrue(extra.isEmpty(), locale + " has keys English does not: " + extra);
        }
    }

    private static Map<String, String> read(String locale) {
        String resource = "/assets/caustica/lang/" + locale + ".json";
        try (InputStream stream = CausticaLangTest.class.getResourceAsStream(resource)) {
            assertNotNull(stream, "missing language file " + resource);
            Map<String, String> strings = new Gson().fromJson(
                    new InputStreamReader(stream, StandardCharsets.UTF_8),
                    new TypeToken<LinkedHashMap<String, String>>() {
                    }.getType());
            assertNotNull(strings, resource + " is not a JSON object");
            return strings;
        } catch (Exception e) {
            throw new AssertionError("could not read " + resource, e);
        }
    }
}
